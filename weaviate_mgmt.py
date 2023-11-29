import sys, os

import weaviate
import weaviate.classes as wvc

from docx import Document

import json

# if (client.collections.exists("Question")):
#     #delete client
#     client.collections.delete("Question")


if __name__ == "__main__":
    if sys.argv[1] == "create_schema_test":
        client = weaviate.WeaviateClient(
            weaviate.ConnectionParams.from_url("http://localhost:8080", 50051)
        )
        if not (client.collections.exists("Question")):
            questions = client.collections.create(
                name="Question",
                vectorizer_config=wvc.Configure.Vectorizer.text2vec_huggingface(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
                # generative_config=wvc.Configure.Generative.openai(),  # Ensure the `generative-openai` module is used for generative queries
                properties=[
                    wvc.Property(
                        name="sop_title",
                        description="SOP Title",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="sop_body",
                        description="SOP Body",
                        data_type=wvc.DataType.TEXT,
                    ),
                ],
            )

    elif sys.argv[1] == "create_schema_upwork":
        client = weaviate.WeaviateClient(
            weaviate.ConnectionParams.from_url("http://localhost:8080", 50051)
        )
        # client.collections.delete("upwork_jobs")
        if not (client.collections.exists("upwork_jobs")):
            upwork = client.collections.create(
                name="upwork_jobs",
                vectorizer_config=wvc.Configure.Vectorizer.text2vec_huggingface(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
                # generative_config=wvc.Configure.Generative.openai(),  # Ensure the `generative-openai` module is used for generative queries
                properties=[
                    wvc.Property(
                        name="job_id",
                        description="Job ID",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="job_title",
                        description="Job Title",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="snippet",
                        description="Job Snippet",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="Category",
                        description="category",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="Subcategory",
                        description="subcategory",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="skills",
                        description="Skills",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="job_duration",
                        description="Job Duration",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="job_type",
                        description="Job Type",
                        data_type=wvc.DataType.TEXT,
                    ),
                    wvc.Property(
                        name="client_country",
                        description="Client Country",
                        data_type=wvc.DataType.TEXT,
                    ),
                ],
            )

    elif sys.argv[1] == "update_record":
        client = weaviate.WeaviateClient(
            weaviate.ConnectionParams.from_url("http://localhost:8080", 50051)
        )
        file = "SOP_CL_XX.XX_PI_Oversite.docx"
        title = file.split(".docx")[0]

        # Load the .docx file
        doc = Document(file)

        full_doc_text = ""

        # Access paragraphs in the document
        for paragraph in doc.paragraphs:
            full_doc_text += "\n" + paragraph.text

        questions = client.collections.get("Question")
        test_update = questions.data.insert(
            {"sop_title": title, "sop_body": full_doc_text}
        )

    elif sys.argv[1] == "update_upwork_records":
        client = weaviate.WeaviateClient(
            weaviate.ConnectionParams.from_url("http://localhost:8080", 50051)
        )
        main_folder = "/Users/robquin/Documents/Professional/Entrepreneur/Bill More Tech/misc/upwork/"
        jobs_folder = main_folder + "jobs/"
        jobs_files = [jobs_folder + x for x in os.listdir(jobs_folder)]

        upwork_jobs = client.collections.get("upwork_jobs")

        client = weaviate.WeaviateClient(
            weaviate.ConnectionParams.from_url("http://localhost:8080", 50051)
        )
        class_name = "Upwork_jobs"
        class_properties = [
            "*"
        ]  # Replace with the specific properties you want to retrieve

        collection = client.collections.get(class_name)

        already_loaded = [item.properties["job_id"] for item in collection.iterator()]
        already_loaded = [str(x) for x in already_loaded]

        print('# already loaded:\t', len(already_loaded))

        counter = 0
        error_counter = 0
        for file in jobs_files:
            # read json file
            with open(file) as f:
                data = json.load(f)

                job_id = data["id"]

                if str(job_id) in already_loaded:
                    pass

                else:
                    title = data["title"]
                    snippet = data["snippet"]
                    category = data["category2"]
                    subcategory = data["subcategory2"]
                    skills = "; ".join(data["skills"])
                    job_duration = data["duration"]
                    job_type = data["job_type"]
                    client_country = data["client"]["country"]

                    # print(json.dumps(data, indent=4))

                    insert_data = {
                        "job_id": job_id,
                        "job_title": title,
                        "snippet": snippet,
                        "category": category,
                        "subcategory": subcategory,
                        "skills": skills,
                        "job_duration": job_duration,
                        "job_type": job_type,
                        "client_country": client_country,
                    }
                    try:
                        upwork_jobs.data.insert(insert_data)
                        counter += 1
                        print("Record loaded:\t", counter)
                    except:
                        error_counter += 1
                        print("error", error_counter)

    elif sys.argv[1] == "show_records":
        client = weaviate.WeaviateClient(
            weaviate.ConnectionParams.from_url("http://localhost:8080", 50051)
        )
        class_name = "Upwork_jobs"
        class_properties = [
            "*"
        ]  # Replace with the specific properties you want to retrieve

        collection = client.collections.get(class_name)

        already_loaded = [item.properties["job_id"] for item in collection.iterator()]

        for item in collection.iterator():
            print(item.properties)

        # Print or process the retrieved objects
        # for obj in response['data']['Get']['YourClassName']:
        #     print(obj)
