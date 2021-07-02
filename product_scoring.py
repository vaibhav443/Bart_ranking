from Bart_scoring_utils import *
import pandas as pd
import time

final_df = pd.read_csv(r"C:\Users\vaibh\Downloads\PM1 Internship\trans_data.csv") #path for the dataset
df = final_df[:100]
data_dict = df.to_dict()

prod_params = {
    "kwds": ["huile bebe"],
    "lang": "fr",
    "data": data_dict,
    "N_prod": 5
}


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


start = time.time()
list_ = product_rank_bart(prod_params)
print(list_)
end = time.time()
print(end - start)
