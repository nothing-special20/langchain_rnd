import re
import sys
import os
import requests
import json
import pandas as pd
import time

from collections import Counter


def env_vars():
    with open(".env", "r") as f:
        lines = f.readlines()

    lines = [re.sub('"|\n', "", x) for x in lines]
    lines = [{x.split("=")[0]: x.split("=")[1]} for x in lines]

    env_object = {}
    for json_ in lines:
        for key, value in json_.items():
            env_object[key] = value

    return env_object


# read the environment variables from the .env file
from dotenv import dotenv_values

# print(load_dotenv())
config = dotenv_values(".env")
RAPID_API_KEY = config["RAPID_API_KEY"]


def get_categories():
    url = "https://upwork-api2.p.rapidapi.com/metadata/categories"

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers)

    data = response.json()["data"]

    subcategory_list = []

    for category in data:
        main_category_name = category["title"]
        for subcat in category["topics"]:
            temp = pd.DataFrame([subcat])
            temp["category"] = main_category_name
            subcategory_list.append(temp)

    subcategory_df = pd.concat(subcategory_list)

    return subcategory_df


def search_freelancers(q, title, skills, rate, offset, count):
    url = "https://upwork-api2.p.rapidapi.com/freelancers"

    querystring = {
        "q": q,  # "laravel",
        "title": title,  # "Full Stack Develoer",
        "skills": skills,  # "laravel",
        "rate": rate,  # "[15 TO 40]",
        "offset": str(offset),  # "0",
        "count": str(count),  # "5"
    }

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)

    try:
        data = response.json()["data"]["freelancersData"]

    except:
        print(response.text)
        data = []

    return data


def write_freelancers(freelancer_data, folder):
    for freelancer in freelancer_data:
        id = freelancer["id"]
        try:
            with open(folder + f"freelancers{id}.json", "w") as f:
                json.dump(freelancer, f, indent=4)
        except:
            print(f"Error writing {id}")


def read_freelancers(folder):
    freelancers = []
    for i in range(1, 11):
        with open(folder + f"freelancers{i}.json", "r") as f:
            freelancers.append(json.load(f))
    return freelancers


def freelancer_profile(id):
    url = f"https://upwork-api2.p.rapidapi.com/freelancers/~{str(id)}"

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers)

    try:
        return response.json()["data"]

    except:
        return {}


def write_freelancer_profile(fl_profile_data, folder):
    try:
        id = fl_profile_data["ciphertext"]

        try:
            with open(folder + f"freelancer_profile_{id}.json", "w") as f:
                json.dump(fl_profile_data, f, indent=4)
        except:
            print(f"Error writing {id}")

    except:
        print("error: no id")


def search_jobs(keyword, count, offset):
    url = "https://upwork-api2.p.rapidapi.com/jobs"

    querystring = {
        "keyword": keyword,  # "marketing",
        "count": str(count),  # "2"
        "offset": str(offset),  # "0",
    }

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "upwork-api2.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)

    try:
        return response.json()["data"]
    except:
        print(response.text)
        return {"jobsData": [], "paging": {}}


def write_jobs(jobs_data, keyword, job_folder, statistics_folder):
    for job in jobs_data["jobsData"]:
        id = job["id"]
        try:
            with open(job_folder + f"job{id}.json", "w") as f:
                json.dump(job, f, indent=4)
        except:
            print(f"Error writing {id}")

    statistics = jobs_data["paging"]
    statistics["keyword"] = keyword

    try:
        with open(
            statistics_folder + f'jobs_statistics_{re.sub("/", "-", keyword)}.json', "w"
        ) as f:
            json.dump(statistics, f, indent=4)
    except:
        print(f"Error writing {keyword}")


def read_jobs(folder):
    jobs = []
    jobs_files = [folder + x for x in os.listdir(folder) if x.endswith(".json")]
    for file in jobs_files:
        with open(file) as f:
            jobs.append(json.load(f))
    return jobs


def read_freelancers(folder):
    jobs = []
    jobs_files = [folder + x for x in os.listdir(folder) if x.endswith(".json")]
    for file in jobs_files:
        with open(file) as f:
            jobs.append(json.load(f))
    return jobs


def skills_list(freelancer_data):
    skills = []
    for freelancer in freelancer_data:
        try:
            skills.extend(freelancer["skills"])
        except:
            print("Error with skills")

    skills = list(set(skills))
    skills.sort()

    return skills


if __name__ == "__main__":
    main_folder = (
        "/Users/robquin/Documents/Professional/Entrepreneur/Bill More Tech/misc/upwork/"
    )
    if sys.argv[1] == "categories":
        # DevOps & Solution Architecture
        test = get_categories()
        subcats = test["title"].tolist()
        counter = 0
        for x in subcats:
            print(x, counter)
            counter += 1

    elif sys.argv[1] == "search_freelancers":
        categories = get_categories()
        # subcats = categories["title"].tolist()
        subcats = ["artificial intelligence", "machine learning"]
        for subcat in subcats:
            print(subcat)
            q = subcat  #'marketing'
            title = ""
            skills = ""
            rate = "[15 TO 400]"
            offset = 0
            count = 50
            for offset in range(0, 100):
                # print(json.dumps(test, indent=4))
                test = search_freelancers(q, title, skills, rate, 50 * offset, count)
                print(q, "offset: ", 50 * offset, "count: ", count)
                write_freelancers(test, main_folder + "freelancers/")
                time.sleep(5)

    elif sys.argv[1] == "freelancer_profile":
        freelancer_ids = [
            x.split("~")[1].split(".")[0]
            for x in os.listdir("data/freelancers/")
            if ".json" in x
        ]
        already_loaded = [
            x.split("~")[1].split(".")[0]
            for x in os.listdir("data/freelancer_profiles/")
            if ".json" in x
        ]
        freelancer_ids = [x for x in freelancer_ids if x not in already_loaded]

        for id in freelancer_ids:
            test = freelancer_profile(id)
            write_freelancer_profile(test, main_folder + "freelancer_profiles/")

    elif sys.argv[1] == "search_jobs":
        categories = get_categories()
        subcats = categories["title"].tolist()

        subcat = sys.argv[2]

        if len(sys.argv) > 3:
            job_start_range = sys.argv[3]
        else:
            job_start_range = 0

        for page in range(job_start_range, 200):
            print("Page ", page)
            test = search_jobs(subcat, 50, 50 * page)
            write_jobs(
                test, subcat, main_folder + "jobs/", main_folder + "job_statistics/"
            )
            time.sleep(5)

        # for subcat in subcats[38:]:
        #     for page in range(0, 100):
        #         print(subcat, page)
        #         test = search_jobs(subcat, 50, 50 * page)
        #         write_jobs(test, subcat, main_folder + 'jobs/', main_folder + 'job_statistics/')
        #         time.sleep(5)

    elif sys.argv[1] == "analyze_jobs":
        job_data = read_jobs("data/jobs/")
        job_data = [
            x
            for x in job_data
            if x["client"]["country"] is not None
            and "United States" in x["client"]["country"]
        ]
        job_data = [
            x
            for x in job_data
            if x["client"]["payment_verification_status"] == "VERIFIED"
        ]
        job_data = [x for x in job_data if x["budget"] > 500]

        job_snippets = [re.sub("\n", " ", x["snippet"]) for x in job_data]
        jobs_titles = [x["title"] for x in job_data]

        job_detail_list = []
        for x in job_data:
            for skill in x["skills"]:
                temp = {"snippet": x["snippet"], "budget": x["budget"], "skills": skill}
                job_detail_list.append(temp)

        job_df = [pd.DataFrame([x]) for x in job_detail_list]
        job_df = pd.concat(job_df)

        grouped_df = job_df.groupby("skills")

        # aggregate functions
        mean_budget = grouped_df["budget"].mean()
        job_count = grouped_df.size()
        sum_budget = grouped_df["budget"].sum()

        # Combine these Series into a new DataFrame
        result_df = pd.DataFrame(
            {
                "mean_budget": mean_budget,
                "job_count": job_count,
                "sum_budget": sum_budget,
            }
        )

        # Now sort by mean_rate in descending order
        result_df = result_df.sort_values(by="job_count", ascending=False).reset_index()

        print(result_df.head(25))

    elif sys.argv[1] == "analyze_freelancers_skills":
        freelancer_data = read_freelancers("data/freelancers/")
        freelancer_data = [
            x
            for x in freelancer_data
            if x["country"] is not None and "United States" in x["country"]
        ]
        freelancer_data = [
            x
            for x in freelancer_data
            if x["portfolio_items_count"] > 10 and x["rate"] > 30
        ]

        skills = skills_list(freelancer_data)
        print(len(skills))

        freelancer_detail_list = []
        for x in freelancer_data:
            if x["skills"] is not None:
                for skill in x["skills"]:
                    temp = {
                        "rate": x["rate"],
                        "skills": skill,
                        "portfolio_items_count": x["portfolio_items_count"],
                    }
                    freelancer_detail_list.append(temp)

        freelancer_detail_df = [pd.DataFrame([x]) for x in freelancer_detail_list]
        freelancer_detail_df = pd.concat(freelancer_detail_df)

        # freelancer_detail_df = freelancer_detail_df.groupby(['skills']).mean('rate').reset_index()
        # freelancer_detail_df = freelancer_detail_df.sort_values(by=['rate'], ascending=False)

        # print(freelancer_detail_df.head(100))

        # for x in skills[0:100]:
        #     print(x)

        # group by skill, get mean rate, get skill count, sort by rate descending
        # First, we need to create a grouped DataFrame
        print(freelancer_detail_df.columns)
        grouped_df = freelancer_detail_df.groupby("skills")

        # Now let's calculate mean rates and skill counts
        mean_rate = grouped_df["rate"].mean()
        skill_count = grouped_df.size()
        mean_portfolio_items_count = grouped_df["portfolio_items_count"].mean()
        sum_portfolio_items_count = grouped_df["portfolio_items_count"].sum()

        # Combine these Series into a new DataFrame
        result_df = pd.DataFrame(
            {
                "mean_rate": mean_rate,
                "skill_count": skill_count,
                "mean_portfolio_items_count": mean_portfolio_items_count,
                "sum_portfolio_items_count": sum_portfolio_items_count,
            }
        )

        # Now sort by mean_rate in descending order
        result_df = result_df.sort_values(by="mean_rate", ascending=False).reset_index()

        result_df = result_df[result_df["skill_count"] > 20]

        result_df = result_df[result_df["mean_rate"] < 70]

        print(result_df.head(9))
        print(result_df.shape)

    elif sys.argv[1] == "analyze_job_skills":
        job_data = read_jobs("data/jobs/")
        job_data = [
            x
            for x in job_data
            if x["client"]["country"] is not None
            and "United States" in x["client"]["country"]
        ]
        job_data = [
            x
            for x in job_data
            if x["client"]["payment_verification_status"] == "VERIFIED"
        ]
        job_data = [x for x in job_data if x["budget"] > 200]
        job_skills = []

        for job in job_data:
            try:
                job_skills.extend(job["skills"])
            except:
                print("Error with skills")

        # job_skills = list(set(job_skills))
        # job_skills.sort()
        most_common_skills = Counter(job_skills).most_common(25)
        print(most_common_skills)

        # filtered_jobs = [x for x in job_data if 'microsoft-excel' in x['skills']]
        # filtered_jobs = [x for x in job_data if 'data-entry' in x['skills']]
        # filtered_jobs = [x for x in job_data if 'marketing-research' in x['skills']]
        # filtered_jobs = [x for x in job_data if 'seo-keyword-research' in x['skills']]
        # filtered_jobs = [x for x in job_data if 'google-analytics' in x['skills']]
        # filtered_jobs = [x for x in job_data if 'lead-generation' in x['skills']]
        filtered_jobs = [x for x in job_data if "api-integration" in x["skills"]]

        filtered_job_desc = [x["snippet"] for x in filtered_jobs]
        filtered_jobs_titles = [
            {"id": x["id"], "title": x["title"]} for x in filtered_jobs
        ]

        for x in filtered_jobs_titles:
            print("\n\n~~~~~~~~")
            print(x)

        print(len(filtered_jobs_titles))

    elif sys.argv[1] == "search_freelancers_by_skill":
        freelancer_data = read_freelancers("data/freelancers/")
        freelancer_data = [
            x
            for x in freelancer_data
            if x["country"] is not None and "United States" in x["country"]
        ]
        freelancer_data = [
            x
            for x in freelancer_data
            if x["portfolio_items_count"] > 10 and x["rate"] > 30
        ]

        skills = skills_list(freelancer_data)
        print(len(skills))

        for skill in skills:
            print(skill)
            q = ""  #'marketing'
            title = ""
            rate = "[15 TO 400]"
            offset = 0
            count = 50
            for offset in range(0, 20):
                # print(json.dumps(test, indent=4))
                test = search_freelancers(q, title, skill, rate, 50 * offset, count)
                print(q, "offset: ", 50 * offset, "count: ", count)
                write_freelancers(test, "data/freelancers/")
                time.sleep(5)

    elif sys.argv[1] == "search_jobs_by_skill":
        freelancer_data = read_freelancers("data/freelancers/")
        freelancer_data = [
            x
            for x in freelancer_data
            if x["country"] is not None and "United States" in x["country"]
        ]
        freelancer_data = [
            x
            for x in freelancer_data
            if x["portfolio_items_count"] > 10 and x["rate"] > 30
        ]

        skills = skills_list(freelancer_data)
        print(len(skills))

        for skill in skills:
            for page in range(0, 20):
                print(skill, page)
                test = search_jobs(skill, 50, 50 * page)
                write_jobs(test, skill, "data/jobs/", "data/job_statistics/")
                time.sleep(5)
