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
        dict_res["top_n_results"] = results[['score', 'title', 'description', 'brand']].to_dict()

        list_res.append(dict_res)

    return list_res

```
Input Format-

Input is a dictionary containing list of keywords,lang, data and No of top products to search.
```python
prod_params = {
    "kwds": ["gélules de vitamine"],
    "lang": "fr",
    "data": data_dict,
    "N_prod": 5
}
```
data_dict is a dictionary containing title, description, brand and price of products.

sample format of data_dict-
```python
{
    "title": {"0":"Alphanova - Lait solaire Bébé bio SPF 50+ - 50 ml - Solaires",
      "1":"Phyto-Actif - Acérola Plus 500 - 2 x 15 comprimés - Vitamines et minéraux",
      },
      "description":{"0":"text of description",
      "1":"text of description"
      },
      "brand":{"0":"Alphanova",
      "1":"Phyto-Actif"},
      "price":{"0":13.25,
      "1":8.9}
}
```

Output format-
Output is a list of dictionaries containing keyword and top_n_results.
```python
[
   {
      "kwd_value":"gélules de vitamine",
      "top_n_results":
         {
            "score":{
               "0":94.7544252872467,
               "1":86.8496572971344,
               "2":62.51157820224762,
               "3":60.15981961041689,
               "4":53.921242356300354
            },
            "title":{
               "0":"Biotechnie - Calcium Marin - 40 gélules - Vitamines et minéraux",
               "1":"Alphanova - Après-soleil Aloé vera et Grenade bio - 125 ml - Solaires",
               "2":"Alphanova - Lait solaire Bébé bio SPF 50+ - 50 ml - Solaires",
               "3":"Phyto-Actif - Acérola Plus 500 - 2 x 15 comprimés - Vitamines et minéraux",
               "4":"Nature's Plus - SOURCE DE VIE Adulte 60 Comprimés - Vitamines et minéraux"
            },
            "description":{
               "0":"text of description",
               "1":"text of description",
               "2":"text of description",
               "3":"text of description",
               "4":"text of description"
            },
            "brand":{
            "0":"Biotechnie",
            "1":"Alphanova",
            "2":"Alphanova",
            "3":"Phyto-Actif",
            "4":"Nature's Plus"
            }
         }
     }
]
```
