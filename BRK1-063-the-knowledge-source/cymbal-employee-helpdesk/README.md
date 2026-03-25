# Sample data for Google Cloud Next Breakout session BRK1-063

This directory contains the sample data used for the **"Build a no-code universal employee helpdesk app with Vertex AI search"** codelab (Session BRK1-063:The knowledge source: Reliable product docs for AI agents via MCP) at Google Cloud Next '26.

## Overview

The demo guides users through building a unified employee portal using Vertex AI Search. This application leverages Retrieval Augmented Generation (RAG) to provide conversational, grounded answers by synthesizing information from both structured financial records and unstructured HR documents.

## Directory structure

The files are organized by data type and destination service:

*   **`HR/`**: Contains **unstructured data**, including `.doc`, `.txt`, and `.html` files. These files represent corporate policies and are intended to be uploaded to a Cloud Storage bucket.
  
*   **`Finance/`**: Contains **structured data** in `.jsonl` (Newline delimited JSON) format. These files, such as `cymbal_employee_finance.jsonl` and `product_inventory.jsonl`, are designed to be imported into **BigQuery** tables.

## How to use these files

To replicate the demo, follow the steps outlined in the [Universal Employee Helpdesk Codelab](https://codelabs.devsite.corp.google.com/next26/universal-employee-helpdesk).

1.  **Clone this repository** to your local machine or Cloud Shell.
2.  **Set up data sources**:
    *   Upload the contents of the `HR/` folder to a **Cloud Storage** bucket.
    *   Load the `.jsonl` files from the `Finance/` folder into a **BigQuery** dataset.
3.  **Configure data stores**: Create Vertex AI Search data stores connected to your Cloud Storage and BigQuery sources.
4.  **Connect and test**: Link the data stores to a **Vertex AI Search app** and interact with the interface to verify synthesized answers.
