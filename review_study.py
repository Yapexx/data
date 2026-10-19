"""Review existing study artifacts without running a study or calculating indicators.

Usage: python -B -m scripts.review_study INPUT_DIRECTORY --output-dir NEW_DIRECTORY
Also accepts a feature_path[_df].csv/.pickle file for summary-only review.
Only load trusted local pickle files. CSV artifacts are supported without pickle.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant_engine.analysis.research_triage import (
    ReviewPolicy, clean, row_profiles, summary_profiles, classify, render_review, profile_tables,
)


def load_study(path,*,summary_only=False,features_path=None,partition='train',combinations_path=None):
    path=Path(path)
    root=path if path.is_dir() else path.parent
    consumed=[]
    def read(file):
        consumed.append(file)
        if file.suffix=='.pickle':
            return pd.read_pickle(file)
        if file.suffix=='.csv':
            return pd.read_csv(file)
        raise ValueError('Expected .pickle or .csv')
    def find(names):
        for name in names:
            for ext in ('.pickle','.csv'):
                file=root/(name+ext)
                if file.exists():
                    return file
        return None
    metadata={}
    meta_file=root/'metadata.json'
    if meta_file.exists():
        consumed.append(meta_file)
        metadata=json.loads(meta_file.read_text(encoding='utf-8'))
    definitions=metadata
    while 'features' not in definitions and isinstance(definitions.get('input_metadata'),dict):
        definitions=definitions['input_metadata']
    path_file=path if path.is_file() else find(('feature_path_df','feature_path'))
    raw_file=None if summary_only or path.is_file() else find(('payoffs','event_payoffs_df','research_df','research'))
    notes=[]
    combinations=None
    if combinations_path:
        if raw_file is None:
            raise ValueError('Combinations require saved row-level observations, not marginal summary tables')
        file=Path(combinations_path)
        consumed.append(file)
        combinations=json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(combinations,dict) or not combinations:
            raise ValueError('Supply a nonempty named combinations object')
    if raw_file is not None:
        frame=read(raw_file)
        if features_path:
            file=Path(features_path)
            consumed.append(file)
            specs=json.loads(file.read_text(encoding='utf-8'))
        elif isinstance(definitions.get('features'),dict):
            specs=definitions['features']
        elif path_file is not None:
            table=read(path_file)
            bins_file=find(('feature_bins_df','feature_bins'))
            bins=read(bins_file) if bins_file else table
            specs={}
            for feature in table.feature.unique():
                if feature not in frame:
                    raise ValueError(f'Declared feature missing in rows: {feature}')
                numeric=pd.api.types.is_numeric_dtype(frame[feature]) and not pd.api.types.is_bool_dtype(frame[feature])
                if numeric:
                    values=bins.loc[bins.feature.eq(feature),['bucket','bin_lower','bin_upper']].drop_duplicates().dropna()
                    values=values.sort_values('bin_lower')
                    if values.empty or not np.array_equal(values.bin_upper.iloc[:-1],values.bin_lower.iloc[1:]):
                        raise ValueError(f'No unambiguous contiguous saved bins for numeric feature {feature}')
                    specs[feature]=dict(description=feature,edges=[values.bin_lower.iloc[0],*values.bin_upper],labels=values.bucket.astype(str).tolist())
                else:
                    specs[feature]=dict(description=feature,edges=None)
        else:
            raise ValueError('No explicit feature definitions: supply --features JSON or a feature_path table; no automatic numeric-column search')
        specs={name:{**spec,'edges':None if spec.get('edges') is None else [float(v) for v in spec['edges']]}
               for name,spec in specs.items()}
        if not specs:
            raise ValueError('Empty feature allowlist')
        profiles,more=row_profiles(frame,specs,partition=partition,combinations=combinations)
        notes += [f'Row-level input: {raw_file.name}. Original bins preserved; no study rerun.']+more
    elif path_file is not None:
        table=read(path_file)
        year_file=find(('yearly_feature_path_df','yearly_feature_path'))
        yearly=read(year_file) if year_file else None
        profiles,more=summary_profiles(table,yearly)
        notes += [f'Aggregate input: {path_file.name}.']+more
        # Recover readable definitions where saved; never infer a horizon from a hash.
        config_file=find(('configurations_df','configurations'))
        configs=read(config_file) if config_file else None
        for profile in profiles:
            if configs is not None and 'outcome_config_id' in configs:
                selected=configs.loc[configs.outcome_config_id.eq(profile['context']['outcome_config_id'])]
                if len(selected)==1:
                    profile['definitions']['outcome_definition']=json.dumps(clean(selected.iloc[0].to_dict()))
    else:
        raise ValueError('No compatible saved row-level study or feature_path table found')
    notes += ['No unseen-validation or causal claim: reviewing historical artifacts is exploratory.',
              'If several formats exist, canonical .pickle precedes .csv; input fingerprints list exactly which files were used.']
    fingerprints={str(file.resolve()):hashlib.sha256(file.read_bytes()).hexdigest() for file in dict.fromkeys(consumed)}
    return profiles,notes,fingerprints


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--features',type=Path,help='Explicit feature definitions JSON when absent from the study')
    parser.add_argument('--combinations',type=Path,help='Named bins/conditions combinations JSON; requires saved rows')
    parser.add_argument('--partition',default='train',help='If split exists, review this partition only (default train)')
    parser.add_argument('--min-count',type=int,default=200)
    parser.add_argument('--min-year-count',type=int,default=50)
    parser.add_argument('--min-years',type=int,default=3)
    args=parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    policy=ReviewPolicy(min_count=args.min_count,min_year_count=args.min_year_count,min_years=args.min_years)
    print('Reading saved study outputs only; no outcomes or signals will be recomputed.',flush=True)
    profiles,notes,fingerprints=load_study(args.input,summary_only=args.summary_only,
                                         features_path=args.features,partition=args.partition,combinations_path=args.combinations)
    records=classify(profiles,policy)
    document=render_review(profiles,records,notes,policy)
    payload=dict(policy=asdict(policy),input_sha256=fingerprints,notes=notes,profiles=profiles,assessments=records,
                 reviewer_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__),Path(__file__).resolve().parents[1]/'crypto_quant_engine/analysis/research_triage.py',
                     Path(__file__).resolve().parents[1]/'crypto_quant_engine/analysis/feature_groups.py')})
    tables=profile_tables(profiles) if args.combinations else {}
    # Verify no source artifacts changed while reading/reviewing them.
    for filename,digest in fingerprints.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Input changed during review: {filename}')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'REVIEW.md').write_text(document,encoding='utf-8')
    (args.output_dir/'evidence.json').write_text(json.dumps(clean(payload),indent=2,allow_nan=False),encoding='utf-8')
    for name,table in tables.items():
        table.to_csv(args.output_dir/(name+'.csv'),index=False)
    print(f'{len(profiles)} profiles; {sum(r["status"]=="descriptive lead" for r in records)} descriptive role-level leads.')
    print(f'Read {args.output_dir/"REVIEW.md"}')
    return profiles,records


if __name__=='__main__':
    main()
