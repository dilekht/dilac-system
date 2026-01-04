#!/usr/bin/env python3
"""
Dictionary Parsing Script
==========================

Parses the Dictionary of Contemporary Arabic text file and
generates LMF-compliant XML and JSON databases.

Usage:
    python parse_dictionary.py <input_file> [--output-dir <dir>]
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dilac.parser import DictionaryParser, DiLACLeskPreprocessor
from dilac.lmf_schema import LexicalResource


def main():
    parser = argparse.ArgumentParser(
        description='Parse Dictionary of Contemporary Arabic'
    )
    parser.add_argument(
        'input_file',
        help='Path to dictionary text file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--format',
        choices=['all', 'xml', 'json', 'lesk'],
        default='all',
        help='Output format(s)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing dictionary: {args.input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Parse dictionary
    dict_parser = DictionaryParser()
    resource = dict_parser.parse_file(args.input_file)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("PARSING COMPLETE")
    print("=" * 50)
    print(f"Total entries: {len(resource.entries)}")
    print(f"Total senses: {sum(len(e.senses) for e in resource.entries)}")
    print(f"Total examples: {sum(len(s.examples) for e in resource.entries for s in e.senses)}")
    print()
    
    # Save outputs
    if args.format in ['all', 'xml']:
        xml_path = output_dir / 'dilac_lmf.xml'
        print(f"Saving LMF XML to: {xml_path}")
        resource.save(str(xml_path))
    
    if args.format in ['all', 'json']:
        json_path = output_dir / 'dilac.json'
        print(f"Saving JSON to: {json_path}")
        dict_parser.parse_to_json(args.input_file, str(json_path))
    
    if args.format in ['all', 'lesk']:
        lesk_path = output_dir / 'dilac_lesk.json'
        print(f"Saving Lesk format to: {lesk_path}")
        preprocessor = DiLACLeskPreprocessor()
        preprocessor.export_lesk_format(resource, str(lesk_path))
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
