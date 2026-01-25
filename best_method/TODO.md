# TODO

For reference using `verbatim_rag` see [lecture notebook](https://github.com/tuw-nlp-ie/tuw-nlp-ie-2025WS/blob/main/lectures/09_RAG/09_RAG_with_outputs.ipynb) or the [project repository](https://github.com/KRLabsOrg/verbatim-rag).

1. Duplicate [base implementation](./base_bm25_plus_llm.ipynb) and add parallel queries to LLM models for:
   1. a general topic of the paper,
   2. focused summaries of the sections,
   3. ranking BM25 selection of sentences,
   4. or whatever you would like to set up.
2. Improve the final step to minimize hallucinations or enforce less changes by LLM re-write.
   1. Compare BM25 selection in our model vs Bertscore selection in `verbatim_rag`.
3. Adapt [qualitative review script](./../data_analysis/qualitative_review.ipynb) to display new results.
   1. Write about whether the reference citation and the summary ever contradict each other.
   2. Build a comparison table of the base model against any implemented improvements.
4. Adapt [quantitative review script](./../data_analysis/quantitative_review.ipynb) to display new results.
5. Adapt this into the [README file](./../README.md).