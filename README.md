Document Retieval program. 

Uses different weighting schemes (tfidf) and uses cosine similarity to rank the documents by order of relevance. 

To run: 

python3 ir_engine.py *opts* *output file*

To evaluate: 

python3 eval_ir.py cacm_gold_std.txt *output file from run*

Use -h for more info. 