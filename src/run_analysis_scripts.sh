
echo "Metadata"
cd metadata
python metadata.py
cd ..

echo "Additional analyses"
cd additional_analyses
python direct_repeats.py
python length_segments.py
python nucleotide_enrichment.py
cd ..

echo long_DelVGs
cd long_delvgs
python long_delvgs.py
cd ..

echo Run overall scripts
cd overall_comparision
python compare_expected.py
python compare_mean.py
python general_analyses.py
cd ..

echo "FINISHED"
