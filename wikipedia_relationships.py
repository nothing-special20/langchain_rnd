import sys, os

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import spacy

import requests
from bs4 import BeautifulSoup
import re
import json

import time

import seaborn as sns

from itertools import combinations
from collections import defaultdict
from collections import Counter
from datetime import datetime

import traceback


# Create a graph
def chart_relationships(df):
    G = nx.Graph()

    articles = list(set(df["article_slug"]))

    for article in articles:
        names = df[df["article_slug"] == article]["names"].to_list()
        print(names)
        for person in names:
            G.add_node(person, layer=0)
            G.add_node(article, layer=1)
            G.add_edge(person, article)

    # Custom positioning for the pyramid shape
    pos = {}
    layer_0 = [node for node in G.nodes if G.nodes[node]["layer"] == 0]
    layer_1 = [node for node in G.nodes if G.nodes[node]["layer"] == 1]

    # Position the top layer (people)
    for i, node in enumerate(layer_0):
        pos[node] = (i, 1)

    # Position the bottom layer (articles)
    for i, node in enumerate(layer_1):
        pos[node] = (i, 0)

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        font_size=10,
        font_weight="bold",
    )

    plt.show()

    # return plt


def co_occurrence_matrix(df):
    grouped = df.groupby("article_slug")["names"].apply(list)

    # Create a default dictionary to store co-occurrences
    co_occurrences = defaultdict(int)

    # Iterate over grouped DataFrame
    for names in grouped:
        # Create all possible combinations of pairs from the list of names
        for pair in combinations(names, 2):
            pair = tuple(sorted(pair))
            co_occurrences[pair] += 1

    # Convert the dictionary to a DataFrame
    co_occurrence_df = pd.DataFrame(
        ((pair[0], pair[1], count) for pair, count in co_occurrences.items()),
        columns=["Name1", "Name2", "Count"],
    )

    min_threshold = 225
    co_occurrence_df = co_occurrence_df[co_occurrence_df["Count"] >= min_threshold]

    # co_occurrence_df = co_occurrence_df.head(1000)

    # pivot_df = co_occurrence_df.pivot('Name1', 'Name2', 'Count').fillna(0)
    # Corrected pivot command
    pivot_df = co_occurrence_df.pivot(
        index="Name1", columns="Name2", values="Count"
    ).fillna(0)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    # sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt='g')
    sns.heatmap(pivot_df, cmap="coolwarm", fmt="g")
    plt.title("Co-occurrence Heatmap of Names")
    plt.xticks(
        rotation=45, ha="right"
    )  # Rotate labels and align right for better readability

    # Optionally, adjust the bottom margin to ensure labels are not cut off
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def scrape_wiki_details(article_slug):
    wiki_base_url = "https://en.wikipedia.org/wiki/"

    url = wiki_base_url + article_slug
    page = requests.get(url)

    # Scrape webpage
    soup = BeautifulSoup(page.content, "html.parser")

    text = soup.find(id="mw-content-text").get_text()

    text = re.sub("\n", " ", text)

    return text


def analyze_wiki_article(article_text, article_slug, article_names_folder):
    nlp = spacy.load("en_core_web_md")
    # Process the text
    # text = "Adolf Hitler 20 April 1889 - 30 April 1945) was an Austrian-born German politician who was the dictator of Germany from 1933 until his suicide in 1945. He rose to power as the leader of the Nazi Party,[a] becoming the chancellor in 1933 and then taking the title of Führer und Reichskanzler in 1934.[b] During his dictatorship, he initiated World War II in Europe by invading Poland on 1 September 1939. He was closely involved in military operations throughout the war and was central to the perpetration of the Holocaust, the genocide of about six million Jews and millions of other victims."
    doc = nlp(article_text)

    # Extract the names
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    names = list(set(names))

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    locations = [ent.text for ent in doc.ents if ent.label_ == "LOC"]

    events = [ent.text for ent in doc.ents if ent.label_ == "EVENT"]

    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    laws = [ent.text for ent in doc.ents if ent.label_ == "LAW"]

    norps = [ent.text for ent in doc.ents if ent.label_ == "NORP"]

    data = {
        "url": url,
        "article_slug": article_slug,
        "article_outer_html": str(soup),
        "article_text": article_text,
        "names": names,
        "dates": dates,
        "locations": locations,
        "events": events,
        "organizations": organizations,
        "laws": laws,
        "norps": norps,
    }
    # turn data into a data frame, where each name has its own record

    # data = pd.DataFrame(data)

    export_file = article_names_folder + article_slug + ".json"
    # data.to_csv(export_file, sep="|", index=False)
    # export to json
    with open(export_file, "w") as fp:
        json.dump(data, fp)


def analyzed_wiki_data(index=None, fields=["article_slug", "names"]):
    files = os.listdir(article_names_folder)
    files = [article_names_folder + file for file in files if file.endswith(".json")]

    if index is not None:
        files = files[:index]

    data = []

    for file in files:
        # temp = pd.read_csv(file, sep="|")
        # read json file to a json object
        with open(file) as json_file:
            _temp = json.load(json_file)
            # only include the article slug and names
            _temp = {k: v for k, v in _temp.items() if k in fields}

        # convert json object to a dataframe
        temp = pd.DataFrame(_temp)

        # concatenate the names into a single record that is a list of all rows
        # temp = temp.groupby('article_slug')['names'].apply(list).reset_index(name='names')
        data.append(temp)

    df = pd.concat(data)
    df.drop_duplicates(inplace=True)

    return df


def create_similarity_matrix(df):
    # Mapping names to articles
    name_to_articles = defaultdict(set)
    for _, row in df.iterrows():
        name_to_articles[row["names"]].add(row["article_slug"])

    print(name_to_articles)
    # List of unique articles
    articles = list(set(df["article_slug"]))

    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((len(articles), len(articles)))

    # Populate the matrix with counts of shared names
    for i, article1 in enumerate(articles):
        for j, article2 in enumerate(articles):
            if i != j:
                shared_names = name_to_articles[article1].intersection(
                    name_to_articles[article2]
                )
                similarity_matrix[i][j] = len(shared_names)

    print(similarity_matrix)
    print(articles)

    return similarity_matrix, articles


def plot_mds(similarity_matrix, articles):
    # Convert the similarity matrix to a distance matrix
    max_similarity = np.max(similarity_matrix)
    distance_matrix = max_similarity - similarity_matrix

    # Apply MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(distance_matrix)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], marker="o")
    for i, article in enumerate(articles):
        plt.text(coords[i, 0], coords[i, 1], article, fontsize=9)
    plt.title("Article Clustering Based on Shared Names")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True)
    plt.show()


def plot_pca(similarity_matrix, articles):
    # Scale the similarity matrix to enhance differences
    scaler = MinMaxScaler()
    scaled_similarity_matrix = scaler.fit_transform(similarity_matrix)

    # Apply PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_similarity_matrix)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], marker="o")
    for i, article in enumerate(articles):
        plt.text(coords[i, 0], coords[i, 1], article, fontsize=9)
    plt.title("Article Clustering with PCA Based on Shared Names")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True)
    plt.show()


def find_unique_sets(x):
    output = list(set(x))
    # sort list by descending
    output.sort()
    output = [str(x) for x in output]
    output = "~|~".join(output)
    return output


def date_cleaning(date_str):
    date_str = re.sub("'", "", date_str)
    date_str = re.sub('"', "", date_str)
    date_str = re.sub(",", "", date_str)
    # hacky but hey it works for now
    date_str = date_str.split(".")[0]

    return date_str


def detect_date_format(date_str):
    # Define regex patterns for various date formats
    patterns = {
        "%Y-%m-%d": r"\d{4}-\d{2}-\d{2}",
        "%d/%m/%Y": r"\d{2}/\d{2}/\d{4}",
        # %d/%m/%Y - MM-DD-YYYY
        # %m/%d/%Y - MM-DD-YYYY
        "%m/%d/%Y": r"\d{2}-\d{2}-\d{4}",
        "YYYY/MM/DD": r"\d{4}/\d{2}/\d{2}",
        # "%d %B %Y": r"\d{2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}",
        "%d %B %Y": r"(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})",
        "%B %d %Y": r"(?:January|February|March|April|May|June|July|August|September|October|November|December) \b\d{2} \d{4}",
        # Add more patterns as needed
    }

    # Test the string against each pattern
    for format, pattern in patterns.items():
        if re.match(pattern, date_str):
            try:
                cleaned_string = re.findall(pattern, date_str)[0]
                if len(cleaned_string) < 2:
                    print("|" + cleaned_string + "|")
                    print("|" + date_str + "|")
                    print(pattern)

                return datetime.strptime(cleaned_string, format)
            except:
                print("error with ")
                print("|" + cleaned_string + "|")
                print("|" + date_str + "|")
                print(pattern)
                print(re.findall(pattern, date_str))
                print(traceback.format_exc())
                return None

    return None


if __name__ == "__main__":
    main_folder = (
        "/Users/robquin/Documents/Professional/Entrepreneur/Bill More Tech/misc/"
    )
    article_names_folder = main_folder + "wiki_search_project/"
    analysis_folder = main_folder + "wiki_analysis/"

    if sys.argv[1] == "chart":
        df = analyzed_wiki_data()

        # only include the top 10 most mentioned names
        # df = df.groupby('names').filter(lambda x: len(x) > 7)
        # df = df[[x in ['Adolf Hitler', 'Hitler', 'Joseph Goebbels'] for x in df['names']]]
        # filter df to only include names with a space
        df = df[[x.find(" ") > 0 for x in df["names"]]]

        df["names"] = [re.sub("'s", "", x) for x in df["names"]]

        # create a new data frame called name count which counts the number of times each name appears
        name_count = df.groupby("names").count().reset_index()
        name_count = name_count.sort_values(by="article_slug", ascending=False)
        name_count = name_count.head(7)

        df = df[[x in name_count["names"].to_list() for x in df["names"]]]

        article_count = df.groupby("article_slug").count().reset_index()
        # sort name count by the number of times each name appears, descending
        article_count = article_count.sort_values(by="names", ascending=False)
        article_count = article_count.head(7)

        # filter df to only include names mentioned in name_count

        df = df[
            [x in article_count["article_slug"].to_list() for x in df["article_slug"]]
        ]

        chart_relationships(df)  # , 'article_slug', 'names'

    elif sys.argv[1] == "get_article_names":
        article_slug = sys.argv[2]
        # Load the language model
        article_text = analyze_wiki_article(article_slug)
        scrape_wiki_details(article_text, article_slug, article_names_folder)

    elif sys.argv[1] == "wiki_scraping":
        # Get URL
        url = "https://en.wikipedia.org/wiki/Adolf_Hitler"
        page = requests.get(url)

        # Scrape webpage
        soup = BeautifulSoup(page.content, "html.parser")

        text = soup.find(id="mw-content-text").get_text()

    elif sys.argv[1] == "spacy_test":
        text = """was an Austrian-born German politician who was the dictator of Germany from 1933 until his suicide in 1945. He rose to power as the leader of the Nazi Party,[a] becoming the chancellor in 1933 and then taking the title of Führer und Reichskanzler in 1934.[b] During his dictatorship, he initiated World War II in Europe by invading Poland on 1 September 1939. He was closely involved in military operations throughout the war and was central to the perpetration of the Holocaust, the genocide of about six million Jews and millions of other victims.

Hitler was born in Braunau am Inn in Austria-Hungary and was raised near Linz. He lived in Vienna later in the first decade of the 1900s before moving to Germany in 1913. He was decorated during his service in the German Army in World War I. In 1919, he joined the German Workers' Party (DAP), the precursor of the Nazi Party, and in 1921 was appointed leader of the Nazi Party. In 1923, he attempted to seize governmental power in a failed coup in Munich and was sentenced to five years in prison, serving just over a year of his sentence. While there, he dictated the first volume of his autobiography and political manifesto Mein Kampf ("My Struggle"). After his early release in 1924, Hitler gained popular support by attacking the Treaty of Versailles and promoting pan-Germanism, anti-Semitism and anti-communism with charismatic oratory and Nazi propaganda. He frequently denounced international capitalism and communism as part of a Jewish conspiracy."""
        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)

        for ent in doc.ents:
            print("~~~~~~~~")
            print(ent.text, ent.label_)

        labels = [x.label_ for x in doc.ents]
        labels = list(set(labels))

        for x in labels:
            print(x)

    elif sys.argv[1] == "rescrape_wiki_articles":
        articles = os.listdir(article_names_folder)
        articles = [x for x in articles if x.endswith(".json")]
        articles = [x.split(".csv")[0] for x in articles]

        for article_slug in articles:
            print(article_slug)
            article_text = analyze_wiki_article(article_slug)
            scrape_wiki_details(article_text, article_slug, article_names_folder)

    elif sys.argv[1] == "update_wiki_article_analysis":
        articles = os.listdir(article_names_folder)
        articles = [x for x in articles if x.endswith(".json")]
        articles = [x.split(".csv")[0] for x in articles]

        for article_slug in articles[:1]:
            print(article_slug)
            article_text = analyze_wiki_article(article_slug)
            scrape_wiki_details(article_text, article_slug, article_names_folder)

    elif sys.argv[1] == "scrape_new_wiki_articles":
        base_url = "https://en.wikipedia.org/wiki/"
        article_slug = sys.argv[2]
        page = requests.get(base_url + article_slug)

        # Scrape webpage
        soup = BeautifulSoup(page.content, "html.parser")

        links = [x.get("href") for x in soup.find_all("a")]

        links = [x for x in links if "/wiki/" in str(x)]
        links = [x for x in links if ":" not in str(x)]

        links = [x.split("/wiki/")[1] for x in links]

        links = list(set(links))

        already_downloaded = os.listdir(article_names_folder)
        already_downloaded = [x.split(".json")[0] for x in already_downloaded]

        print(len(links))
        links = [x for x in links if x not in already_downloaded]
        print(len(links))

        # for link in links:
        #     print(link)

        for article_slug in links:
            print(article_slug)
            try:
                article_text = analyze_wiki_article(article_slug)
                scrape_wiki_details(article_text, article_slug, article_names_folder)
            except:
                print("error with " + article_slug)

    elif sys.argv[1] == "co_occurrence_matrix":
        df = analyzed_wiki_data()
        df = df[[x.find(" ") > 0 for x in df["names"]]]
        co_occurrence_matrix(df)

    elif sys.argv[1] == "test_thing":
        df = analyzed_wiki_data()
        # filter df to include only article_slug with string container Winston
        df = df[
            [
                x.lower().find("hitler") > 0
                or x.lower().find("crypto") > 0
                or x.lower().find("stalin") > 0
                for x in df["article_slug"]
            ]
        ]
        # number of unique article slugs
        print(len(set(df["article_slug"])))
        print(df.shape)

        df.dropna(inplace=True)
        # Assuming df is your DataFrame
        similarity_matrix, articles = create_similarity_matrix(df)
        # plot_mds(similarity_matrix, articles)
        plot_pca(similarity_matrix, articles)

    elif sys.argv[1] == "set_comprehension":
        df = analyzed_wiki_data()
        df.to_csv(analysis_folder + "main_data_audit.csv", index=False)
        # counter the number of times each name appears
        name_counter = Counter(df["names"])
        # calculate the median from the name_counter
        percentile_threshold = np.percentile(np.array(list(name_counter.values())), 92)

        print(percentile_threshold)

        # filter out names that appear only once
        name_counter = {
            k: v for k, v in name_counter.items() if v > percentile_threshold
        }

        print(len(name_counter))

        # print(name_counter)

        # filter df to only include names that appear more than once
        df = df[[x in name_counter.keys() for x in df["names"]]]

        result_df = (
            df.groupby("names")["article_slug"].apply(find_unique_sets).reset_index()
        )
        # result_df = df.groupby('article_slug')['names'].agg(names=set, count='count').reset_index()
        result_df = result_df.groupby("article_slug")["names"].agg(set).reset_index()
        result_df = result_df[[len(x) > 1 for x in result_df["names"]]]
        result_df = result_df[["~|~" in x for x in result_df["article_slug"]]]
        result_df["article_slug"] = [
            set(x.split("~|~")) for x in result_df["article_slug"]
        ]
        result_df["article_overlap_count"] = [len(x) for x in result_df["article_slug"]]
        result_df["names_count"] = [len(x) for x in result_df["names"]]

        # sort result_df by names_count
        result_df = result_df.sort_values(by="names_count", ascending=False)

        result_df.to_csv(
            analysis_folder + "overlapping_names_and_articles.csv", index=False
        )

        print(result_df.shape)

    elif sys.argv[1] == "find_dates_from_related_articles":
        overlapping_data = pd.read_csv(
            analysis_folder + "overlapping_names_and_articles.csv"
        )

        articles = overlapping_data["article_slug"].to_list()

        counter = 0
        # for x in articles:
        #     if len(x) < 200:
        #         print(counter, x)
        #     counter += 1

        test_article = articles  # [26:28]

        test_article = [re.sub("|\{|\\}|'", "", x) for x in test_article]
        test_article = [x.split(", ") for x in test_article]
        # flatten the list
        test_article = [item for sublist in test_article for item in sublist]

        df = analyzed_wiki_data(index=None, fields=["article_slug", "dates"])
        df = df[[x in test_article for x in df["article_slug"]]]

        df["dates"] = [date_cleaning(x) for x in df["dates"]]
        df["formatted_date"] = [detect_date_format(x) for x in df["dates"]]

        df = df[[x != None for x in df["formatted_date"].to_list()]]
        df = df.dropna()

        # format the date column to be a datetime object using the date_format column

        print(df)
        print(df.shape)
        df.to_csv(analysis_folder + "test_article_dates.csv", index=False)

    elif sys.argv[1] == "test_":
        # date_str = "2023-11-25"
        # print(f"Date: {date_str}, Format: {detect_date_format(date_str)}")

        print(datetime.strptime("3 October 2021", "%d %B %Y"))

        re.findall(
            r"\d{1} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}",
            "3 October 2021",
        )[0]
        pattern = r"(\d{1} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})"
