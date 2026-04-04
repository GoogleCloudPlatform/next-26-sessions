# Polyglot E-Commerce

1. First set up and configure AlloyDB. Create table and insert records (Refer alloydb_insert_queries.sql file).
2. Then set up MongoDB, insert documents from files: product_details_export.json and user_interactions_export.json files.
3. Set up BigQuery dataset and table.
4. Set up Google Cloud Storage Bucket and upload files.
5. Set up Toolbox server, update tools.yaml and deploy to Cloud Run.
6. First run agentengine.py and get the entire Reasoning Engine ID path. You will need to use it in the APP_NAME variable.
7. Fill up values for rest of the .env variables and run the application.

For detailed instructions, you can refer to the blog:
https://medium.com/google-cloud/architecting-for-data-diversity-the-intelligent-e-commerce-catalog-4ceadf4bf104
