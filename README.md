# Bart_ranking

Product ranking algorithm based on bart.
Main function-
```python
def product_rank_bart(prod_params):
    """

    :param prod_params: product parameters
    :return: ranked products with respect to each product
    """
    list_res = []
    Df = pd.DataFrame(prod_params.get("data"))
    Df_ST = basic_preprocessing_ST(Df)
    keyword_ST = basic_preprocessing_keywords_ST(prod_params.get("kwds").copy())
    for index in range(len(keyword_ST)):
        dict_res = {}
        dict_bart = bart_scores(Df_ST, keyword_ST[index], prod_params.get("lang", "en"))
        results = bart_result(dict_bart, prod_params.get("N_prod", 3), Df)
        # print(results)
        dict_res["kwd_value"] = prod_params.get("kwds")[index]
        dict_res["top_n_results"] = [results[['score', 'title', 'description']].to_dict()]

        list_res.append(dict_res)

    return list_res

```
Input Format-
Input is a dictionary containing list of keywords,lang, data and No of top products to search.
```python

```
