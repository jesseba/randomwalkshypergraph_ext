import numpy as np
import pandas as pd
import zipfile
import io
from scipy import sparse

def compute_pr(P, r, n, eps=1e-8):
    x = np.ones(n) / n * 1.0
    flag = True
    t = 0
    while flag:
        x_new = (1 - r) * P @ x
        x_new = x_new + np.ones(n) * r / n
        diff = np.linalg.norm(x_new - x, ord=1)
        if diff < eps and t > 100:
            flag = False
        t += 1
        x = x_new
    return x

def read_and_filter_data(zip_path):
    conferences = [
        "neural information processing systems",
        "international conference on machine learning",
        "knowledge discovery and data mining",
        "international joint conference on artificial intelligence",
        "uncertainty in artificial intelligence",
        "international conference on learning representations",
        "computational learning theory"
    ]
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        
        if len(json_files) != 4:
            raise ValueError("There are not exactly four JSON files in the zip.")
        
        dataframes = []
        for json_file in json_files:
            with z.open(json_file) as f:
                data = pd.read_json(io.BytesIO(f.read()), lines=True)
            dataframes.append(data)
    
    def filter_venues(df):
        df['venue'] = df['venue'].str.lower()
        return df[df['venue'].isin(conferences)]
    
    filtered_dfs = [filter_venues(df) for df in dataframes]
    return pd.concat(filtered_dfs, ignore_index=True)

def is_alphabetically_sorted(authors_list):
    if not authors_list:
        return False
    sorted_authors = sorted(authors_list)
    return authors_list == sorted_authors

def assign_uni_weights(authors):
    if is_alphabetically_sorted(authors):
        return [1] * len(authors)
    else:
        if len(authors) == 1:
            return [1]
        else:
            weights = [1] * len(authors)
            weights[0] = 1
            weights[-1] = 1
            return weights

def generate_citation_list(combined_filtered_df):
    citation_list = []
    for authors in combined_filtered_df['authors']:
        if isinstance(authors, list): 
            weights = assign_uni_weights(authors)
            citation_list.append((authors, weights))
        else:
            citation_list.append((authors, []))
    return citation_list

def build_universe_and_matrices(citation_list, combined_filtered_df):
    universe = set()
    for authors, _ in citation_list:
        universe.update(authors)

    universe = np.array(list(universe))
    pi_list = citation_list

    m = len(pi_list)
    n = len(universe)
    R = np.zeros([m, n])
    W = np.zeros([n, m])

    for i in range(len(pi_list)):
        pi, scores = pi_list[i]
        if len(pi) > 1:
            for j in range(len(pi)):
                v = pi[j]
                v = np.where(universe == v)[0][0]
                R[i, v] = scores[j]
                W[v, i] = combined_filtered_df.iloc[:, 2][i] + 1

            R[i, :] = R[i, :] / sum(R[i, :])

    return universe, pi_list, R, W

def build_probability_transition_matrix(universe, pi_list, combined_filtered_df):
    _, _, R, W = build_universe_and_matrices(pi_list, combined_filtered_df)

    W = np.nan_to_num(W, nan=0.0)

    sum_W = W.sum(axis=1)

    zero_sum_rows = np.where(sum_W == 0)[0]
    nan_sum_rows = np.where(np.isnan(sum_W))[0]

    # in case of 0
    sum_W_corrected = sum_W.copy()
    sum_W_corrected[sum_W_corrected == 0] = 1

    
    Wnorm = W / W.sum(axis=1)[:, None]
    Ws = sparse.csr_matrix(Wnorm)
    Rs = sparse.csr_matrix(R)
    
    P = np.transpose(Ws.dot(Rs))
    return P

def main():

    zip_path = './data/dblp.v10.zip'
    output_file = 'raw_rankings_output_ind.txt' 

    combined_filtered_df = read_and_filter_data(zip_path)

    citation_list = generate_citation_list(combined_filtered_df)

    universe, pi_list, _, _ = build_universe_and_matrices(citation_list, combined_filtered_df)

    P = build_probability_transition_matrix(universe, pi_list, combined_filtered_df)

    r = 0.40
    rankings_hg = compute_pr(P, r, len(universe), eps=1e-8).flatten()


    with open(output_file, 'w') as f:
        for rank in rankings_hg:
            f.write(f"{rank}\n")

    print(f"Raw rankings have been saved to {output_file}")

if __name__ == '__main__':
    main()