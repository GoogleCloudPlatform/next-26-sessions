# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
from datetime import datetime
from google import genai
from google import adk
from google.genai import types as genai_types
from toolbox_core import ToolboxSyncClient
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService
from google.adk.sessions import InMemorySessionService  # Add this import
import asyncio
import threading
import vertexai
from flask_cors import CORS

# Load environment variables (including MONGODB_CONNECTION_STRING and server URL)
load_dotenv()
GCS_BUCKET_NAME = os.getenv('GCS_PRODUCT_BUCKET', 'placeholder-bucket')
GCS_BASE_URL = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}"
FALLBACK_IMAGE_URL = os.getenv('FALLBACK_IMAGE_URL')
mongodb_details = " "
# Initialize the LLM (Gemini is perfect for this)
#google_key = os.getenv("GEMINI_API_KEY")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
PROJECT_ID = os.getenv("PROJECT_ID")
APP_NAME = os.getenv("APP_NAME")
USER = "default_user"
MODEL = os.getenv("MODEL")

# Fix the initialization
#client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client = vertexai.Client(
  project=PROJECT_ID,
  location=GOOGLE_CLOUD_LOCATION
)
model_id = MODEL
# Initialize the MCP Toolbox Client
# This client communicates with the running MCP Toolbox Server (usually on localhost:5000)
TOOLBOX_URL = os.getenv("MCP_TOOLBOX_SERVER_URL")
if not TOOLBOX_URL:
    raise ValueError("MCP_TOOLBOX_SERVER_URL not set in environment.")

try:
    toolboxCore = ToolboxSyncClient(TOOLBOX_URL)
    all_tools = toolboxCore.load_toolset("ecommerce_toolset")
    print(f"-> MCP Client: Successfully loaded {len(all_tools)} tools from {TOOLBOX_URL} (Session Kept Open)")
except Exception as e:
    print(f"FATAL ERROR: Could not connect to MCP Toolbox Server at {TOOLBOX_URL}. Error: {e}")
    exit(1)

# Replace your current get_preloaded_tool function with this:
def get_preloaded_tool(tool_name):
    """Finds and returns a loaded tool from the pre-warmed toolset."""
    for tool in all_tools:
        # Check standard SDK name attribute
        if hasattr(tool, 'name') and tool.name == tool_name:
            return tool
        # Check if the tool uses private property mapping
        if hasattr(tool, '_name') and tool._name == tool_name:
            return tool
        # Check dictionary-like structure
        if isinstance(tool, dict) and tool.get('name') == tool_name:
            return tool
        # Fallback to string representations
        if str(tool) == tool_name:
            return tool
            
    # Print the available tools for debugging if lookups fail
    available_names = []
    for t in all_tools:
        if hasattr(t, 'name'): available_names.append(t.name)
        elif hasattr(t, '_name'): available_names.append(t._name)
        else: available_names.append(str(t))
    print(f"ERROR: Tool '{tool_name}' not found. Available tools: {available_names}")
    raise ValueError(f"Tool '{tool_name}' was not found in the loaded toolset.")

def execute_tool(tool_name, params=None):
    """
    Safely finds a preloaded tool and executes it natively.
    Resolves compatibility between the legacy '.invoke()' and modern callable '__call__' structures.
    """
    tool = get_preloaded_tool(tool_name)
    params = params or {}
    
    print(f"DEBUG: Attempting to execute '{tool_name}' with parameters {params}")
    
    try:
        # 1. Prefer legacy invoke if explicitly supported by this Tool wrapper
        if hasattr(tool, 'invoke') and callable(getattr(tool, 'invoke')):
            print(f"DEBUG: Executing '{tool_name}' using fallback '.invoke()'")
            return tool.invoke(params)
            
        # 2. Modern stateless callable execution fallback
        elif callable(tool):
            print(f"DEBUG: Executing '{tool_name}' directly as a function call")
            # Unpack params dictionary safely
            return tool(**params)
            
        else:
            # Check if we can extract invoke from a private/hidden property
            for attr in ['_invoke', '_run', 'run']:
                if hasattr(tool, attr) and callable(getattr(tool, attr)):
                    print(f"DEBUG: Executing '{tool_name}' using custom attribute '{attr}'")
                    return getattr(tool, attr)(**params)
                    
            raise AttributeError(f"Tool {tool_name} is neither callable nor does it support '.invoke()'")
            
    except Exception as e:
        print(f"ERROR: Failed executing tool '{tool_name}' with parameters {params}. Error: {e}")
        raise e



# Helper function to safely decode data received from the MCP client
def safe_decode_data(data):
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON string: {data[:50]}...")
            return None
    return data 



app = Flask(__name__)
# This allows standard cross-origin resource requests on all endpoints
CORS(app, resources={r"/*": {"origins": "*"}})

# --- New Route to Serve the Frontend ---
@app.route('/')
def index():
    """Renders the main product catalog page."""
    return render_template('index.html')

# --- Routes ---
@app.route('/products/<product_id>', methods=['GET'])
def get_product(product_id):
    raw_core = None
    raw_details = None

    # --- 1. FETCH CORE TRANSACTIONAL DATA (AlloyDB) ---
    try:
        raw_core_response = execute_tool("get_product_core_data", {"product_id": product_id})
        decoded_core = safe_load_tool_result(raw_core_response)
        if decoded_core and isinstance(decoded_core, list):
            raw_core = decoded_core[0]
        elif isinstance(decoded_core, dict):
            raw_core = decoded_core
        print(f"DEBUG /products/<id> Core Data: {raw_core}")
    except Exception as e:
        print(f"Warning: AlloyDB core data fetch failed for ID {product_id}: {e}")

    # --- 2. FETCH FLEXIBLE CATALOG DETAILS (MongoDB) ---
    try:
        raw_details_response = execute_tool("get_product_details", {"product_id": product_id})
        decoded_details = safe_load_tool_result(raw_details_response)
        if decoded_details and isinstance(decoded_details, list):
            raw_details = decoded_details[0]
        elif isinstance(decoded_details, dict):
            raw_details = decoded_details
        print(f"DEBUG /products/<id> Details Data: {raw_details}")
    except Exception as e:
        print(f"Warning: MongoDB detail fetch failed for ID {product_id}: {e}")

    # --- 3. MERGE LOGIC ---
    core_data = raw_core if isinstance(raw_core, dict) else {}
    details_data = raw_details if isinstance(raw_details, dict) else {}

    if core_data or details_data:
        full_product = {**details_data, **core_data}
        if not details_data:
            full_product['source_note'] = 'PARTIAL MODE: MongoDB details missing.'
    else:
        # Fallback: Check if the product was already loaded in memory or catalog
        print(f"ERROR: Both AlloyDB and MongoDB missed for ID {product_id}")
        return jsonify({"message": f"Product ID {product_id} not found in any data store."}), 404

    # Ensure price has a valid numeric value
    if 'price' not in full_product or full_product['price'] is None:
        full_product['price'] = 0.0

    # Ensure stock exists
    if 'stock' not in full_product or full_product['stock'] is None:
        full_product['stock'] = 0

    # --- 4. Final Enrichment (GCS Image URL) ---
    sku = full_product.get('sku', '')
    if sku and sku != 'N/A':
        full_product['image_url'] = f"{GCS_BASE_URL}/{sku}.jpg"
        full_product['fallback_url'] = FALLBACK_IMAGE_URL
    else:
        full_product['image_url'] = FALLBACK_IMAGE_URL
        full_product['fallback_url'] = FALLBACK_IMAGE_URL

    return jsonify(full_product)


@app.route('/inventory/<category>', methods=['GET'])
def get_category_inventory_stats(category):
    """
    Demonstrates using a single MongoDB Aggregation tool for analytics.
    """
    try:
        # execute_tool handles the entire lookup and native call for you!
        stats_data = execute_tool("get_product_stats_by_category", {"category": category})
        
        # Ensure we decode the result safely before sending it to the frontend
        parsed_stats = safe_load_tool_result(stats_data)
        return jsonify({
            "message": "Product statistics successfully aggregated from MongoDB.",
            "statistics": parsed_stats
        })
    except Exception as e:
        print(f"Error on category inventory lookup: {e}")
        return jsonify({"error": "Failed to run category aggregation tool.", "details": str(e)}), 500




def safe_load_tool_result(result):
    if result is None:
        return []

    # --- Step 1: Unpack standard Stateless MCP list wrapper content ---
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if hasattr(first_item, 'text'):
            result = first_item.text
        elif isinstance(first_item, dict) and 'text' in first_item:
            result = first_item['text']

    # Unpack nested structures
    if hasattr(result, 'content'):
        if isinstance(result.content, list) and len(result.content) > 0:
            first_content = result.content[0]
            if hasattr(first_content, 'text'):
                result = first_content.text
            elif isinstance(first_content, dict) and 'text' in first_content:
                result = first_content['text']
        else:
            result = result.content
            
    if hasattr(result, 'text'):
        result = result.text

    # --- Step 2: Parse and Clean JSON strings ---
    if isinstance(result, str):
        result_str = result.strip()
        # Clean standard unescaped Markdown JSON blocks
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        elif result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        result_str = result_str.strip()
        
        try:
            parsed = json.loads(result_str)
            if isinstance(parsed, dict):
                for key in ['result', 'data', 'products', 'content', 'items']:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON string from tool result. Reason: {e}")
            print(f"Failed string context: {result_str[:150]}")
            return []

    # --- Step 3: Parse standard Python Dictionaries ---
    elif isinstance(result, dict):
        for key in ['result', 'data', 'products', 'content', 'items']:
            if key in result and isinstance(result[key], list):
                return result[key]
        return [result]

    # --- Step 4: Fallback for clean lists ---
    elif isinstance(result, list):
        cleaned_list = []
        for item in result:
            if isinstance(item, dict):
                cleaned_list.append(item)
            elif hasattr(item, 'text'):
                try:
                    parsed_item = json.loads(item.text)
                    if isinstance(parsed_item, list):
                        cleaned_list.extend(parsed_item)
                    else:
                        cleaned_list.append(parsed_item)
                except Exception:
                    pass
        if cleaned_list:
            return cleaned_list
        return result

    return []



@app.route('/products', methods=['GET'])
def list_products():
    final_catalog = []
    try:
        # 1. Fetch Core Data (AlloyDB)
        raw_core_res = execute_tool("list_products_core", {})
        alloydb_products = safe_load_tool_result(raw_core_res)
        print(f"DEBUG: AlloyDB raw response: {raw_core_res}")
  
        # 2. Fetch All Rich Details (MongoDB)
        raw_details_res = execute_tool("list_all_product_details", {})
        mongodb_details = safe_load_tool_result(raw_details_res)
        print(f"DEBUG: MongoDB raw response: {raw_details_res}")
  
        details_map = {item['product_id']: item for item in mongodb_details if isinstance(item, dict) and 'product_id' in item}
  
        # 3. Merge Strategy
        for core_prod in alloydb_products:
            if not isinstance(core_prod, dict):
                continue
            pid = core_prod.get('product_id')
            sku = core_prod.get('sku')
            rich_details = details_map.get(pid, {})
            
            merged_product = {**rich_details, **core_prod}
            merged_product['image_url'] = f"{GCS_BASE_URL}/{sku}.jpg" if sku else FALLBACK_IMAGE_URL
            merged_product['source'] = 'Polyglot (AlloyDB + MongoDB)'
            final_catalog.append(merged_product)
  
        # 4. Handle edge case: Products in Mongo but NOT in AlloyDB (Disjoint)
        alloy_ids = {p.get('product_id') for p in alloydb_products if isinstance(p, dict)}
        for pid, details in details_map.items():
            if pid not in alloy_ids:
                details['name'] = f"New: {details.get('category', 'Product')}"
                details['price'] = 0.00
                details['stock'] = 0
                details['source'] = 'MongoDB Only'
                details['image_url'] = f"{GCS_BASE_URL}/{details.get('sku')}.jpg" if details.get('sku') else FALLBACK_IMAGE_URL
                final_catalog.append(details)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error building unified catalog: {e}")
        return jsonify({"error": "Catalog generation failed", "details": str(e)}), 500

    return jsonify(final_catalog)



@app.route('/track/view', methods=['POST'])
def track_user_view():
    """
    Records a user product view event to MongoDB (via MCP Tool).
    This is a high-volume write operation.
    """
    data = request.json
    # Simple validation and default data
    user_id = data.get('user_id', 'User')
    product_id = data.get('product_id')
    event_type = 'product_view'
    
    if not product_id:
        return jsonify({"error": "product_id is required for tracking."}), 400

    try:
        # 1. Load the specific MongoDB insertion tool
        insert_tool = get_preloaded_tool("insert_user_interaction")

        # 2. Prepare the data as a JSON string
        data = {
            "user_id": user_id,
            "product_id": product_id,
            "details": "User viewed this product.",
            "timestamp": datetime.utcnow().isoformat()  # Add timestamp
        }
        data_json = json.dumps(data)

        # 3. Invoke the tool with the data parameter
        response = execute_tool("insert_user_interaction", {"data": data_json})

        # 4. Process the response
        print(response)
        
        return jsonify({
            "message": "Interaction tracked successfully (via MongoDB).",
            "inserted_id": response
        }), 201
        
    except Exception as e:
        print(f"Error while recording user interaction: {e}")
        return jsonify({"error": "Failed to record user interaction.", "details": str(e)}), 500


    
@app.route('/product_by_id', methods=['POST'])
def get_product_by_id():
    """
    Retrieves a single product, prioritizing AlloyDB data and falling back to MongoDB details.
    """
    data = request.json
    user_id = data.get('user_id', 'User')
    product_id = data.get('product_id')
    if not product_id:
        return jsonify({"error": "product_id is required."}), 400

    raw_core = None
    raw_details = None

    # --- 1. FETCH CORE TRANSACTIONAL DATA (AlloyDB) ---
    try:
        raw_core_response = execute_tool("get_product_core_data", {"product_id": product_id})
        # Decode the raw response string/list safely
        decoded_core_list = safe_decode_data(raw_core_response)
        raw_core = decoded_core_list[0] if isinstance(decoded_core_list, list) and decoded_core_list else None
    except Exception as e:
        print(f"Warning: AlloyDB core data fetch failed for ID {product_id}. {e}")

    # --- 2. FETCH FLEXIBLE CATALOG DETAILS (MongoDB) ---
    try:
        raw_details_response = execute_tool("get_product_details", {"product_id": product_id})
        # Decode the raw response string/list safely
        decoded_details_list = safe_decode_data(raw_details_response)
        raw_details = decoded_details_list[0] if isinstance(decoded_details_list, list) and decoded_details_list else None
    except Exception as e:
        print(f"Warning: MongoDB detail fetch failed for ID {product_id}. {e}")

    # --- 3. MERGE AND FALLBACK LOGIC ---
    core_data = {} if not raw_core else raw_core
    details_data = {} if not raw_details else raw_details

    # SCENARIO A: Full Merge OR SCENARIO C (AlloyDB Hit)
    if core_data:
        full_product = {**core_data, **details_data} 
        if not details_data:
            full_product['source_note'] = 'PARTIAL MODE: MongoDB details missing.'
    elif details_data:
        # SCENARIO B: AlloyDB Miss, MongoDB Hit (The Disjoint Fallback)
        synth_core = {
            'product_id': details_data.get('product_id'),
            'name': f"MongoDB Product: {details_data.get('category', 'Generic')}", 
            'price': 39.99, 
            'sku': details_data.get('sku', 'SYNTH-001'), 
            'stock': 999,
            'source_note': 'FALLBACK MODE: Core data synthesized from MongoDB details.'
        }
        full_product = {**synth_core, **details_data}
    else:
        # SCENARIO D: Total Miss
        return jsonify({"message": f"Product ID {product_id} not found in any data store."}), 404

    # --- 4. Final Enrichment (GCS Image URL) ---
    sku = full_product.get('sku', 'N/A')
    if sku and sku != 'N/A':
        full_product['image_url'] = f"{GCS_BASE_URL}/{sku}.jpg"
        full_product['fallback_url'] = FALLBACK_IMAGE_URL
    else:
        full_product['image_url'] = FALLBACK_IMAGE_URL
        full_product['fallback_url'] = FALLBACK_IMAGE_URL
    return jsonify(full_product)



@app.route('/etl/run', methods=['POST'])
def run_etl_to_bigquery():
    """
    Orchestrates the application-driven ETL process:
    1. READ: Aggregates interaction data from MongoDB.
    2. WRITE: MERGES the resulting summary data into BigQuery.
    """
    try:
        # --- 1. READ/EXTRACT/TRANSFORM (MongoDB via MCP) ---
        mongo_summary_tool = get_preloaded_tool("get_total_interactions_count")
        
        # This returns the aggregated list: [{'product_id': '...', 'interaction_count': N}, ...]
        summary_data = execute_tool("get_total_interactions_count", {"product_id":""})
        
        if not summary_data:
            return jsonify({"message": "No interaction data to transfer."}), 200

        # --- 2. WRITE/LOAD (BigQuery via MCP) ---
        bq_write_tool = get_preloaded_tool("execute_sql_tool")
        
        # Hardcoded JSON string - THIS IS THE KEY STEP
        #hardcoded_json_string = '[{"interaction_count":1,"product_id":"06523234-2a5c-49fb-b801-e18b72ee3578"}]'


        # BigQuery tool execution
        bq_response = execute_tool("execute_sql_tool", {"product_summaries": summary_data})
        print(bq_response)
        
        return jsonify({
            "message": "Application-Driven ETL complete. MongoDB summary merged into BigQuery.",
            "products_processed": len(summary_data),
            "bigquery_response": "success" # Contains job/status details
        }), 200

    except Exception as e:
        return jsonify({"error": "ETL orchestration failed.", "details": str(e)}), 500



@app.route('/analytics/top5', methods=['GET'])
def get_top_5_products():
    try:
        # 1. Fetch ranking from BigQuery
        top5_response = execute_tool("get_top_5_views", {})
        if not top5_response:
            return jsonify([]), 200

        items = safe_load_tool_result(top5_response)
        if not items or not isinstance(items, list):
            items = safe_decode_data(top5_response)
            if not items or not isinstance(items, list):
                return jsonify([]), 200

        top_products_list = []
        for top_item in items:
            if not isinstance(top_item, dict):
                continue
            product_id = top_item.get('product_id')
            if not product_id: 
                continue

            # --- 2. HYDRATE CORE (AlloyDB) ---
            core_data = {}
            try:
                core_res = execute_tool("get_product_core_data", {"product_id": product_id})
                decoded_core = safe_load_tool_result(core_res)
                if decoded_core and isinstance(decoded_core, list):
                    core_data = decoded_core[0]
            except Exception as e:
                print(f"AlloyDB fail for {product_id}: {e}")

            # --- 3. HYDRATE DETAILS (MongoDB) ---
            details_data = {}
            try:
                details_res = execute_tool("get_product_details", {"product_id": product_id})
                decoded_details = safe_load_tool_result(details_res)
                if decoded_details and isinstance(decoded_details, list):
                    details_data = decoded_details[0]
            except Exception as e:
                print(f"MongoDB fail for {product_id}: {e}")

            # --- 4. MERGE & ENRICH ---
            if core_data or details_data:
                product = {**details_data, **core_data}
                product['total_views'] = top_item.get('interaction_score', top_item.get('interaction_count', 0))
                sku = product.get('sku')
                product['image_url'] = f"{GCS_BASE_URL}/{sku}.jpg" if sku else FALLBACK_IMAGE_URL
                top_products_list.append(product)

        return jsonify(top_products_list)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Analytics Exception: {e}")
        return jsonify({"error": "BigQuery Analytics failed.", "details": str(e)}), 500


# Catalog Specialist: Handles only MongoDB rich data
catalog_agent = adk.Agent(
    name="CatalogSpecialist",
    model=MODEL,
    description="Expert in product descriptions, reviews, and detailed specs from MongoDB.",
    instruction="""
    Use 'list_all_product_details' for all product data and 'get_product_details' to answer detailed product questions.
    Always summarize customer sentiment from reviews if available.
    """,
    tools=all_tools
)

# Updated InventorySpecialist
inventory_agent = adk.Agent(
    name="InventorySpecialist",
    model=MODEL,
    description="Specialist in real-time stock, pricing, and sales trends.",
    instruction="""
    Use 'get_product_core_data' for stock/price and 'get_top_5_views' for trending data.
    If a user asks for 'Top' or 'Trending' items in a specific category:
    1. Get the trending product IDs using 'get_top_5_views'.
    2. Use 'list_all_product_details' to filter those IDs by the requested category.
    3. If 'get_top_5_views' does not allow filtering, fetch ALL trending items first, then manually filter them using the category info from MongoDB.
    Always return Product Name, SKU, and Price—never the raw Product ID.
    """,
    tools=all_tools
)

# ETL Specialist: Handles MongoDB and BigQuery
etl_agent = adk.Agent(
    name="ETLSpecialist",
    model=MODEL,
    description="Specialist in orchestrating the application-driven ETL process.",
    instruction="""
    Your job is to orchestrate the application-driven ETL process:
    1. READ: Aggregate interaction data from MongoDB (using the tool get_total_interactions_count).
    2. WRITE: MERGE the resulting summary data into BigQuery (using the tool execute_sql).
    
    Use 'get_product_core_data' for stock/price and 'get_top_5_views' for trending data.
    If a user asks for 'Top' items, prioritize BigQuery analytics results.
    """,
    tools=all_tools
)

# ETL Specialist: Handles MongoDB and BigQuery
bi_agent = adk.Agent(
    name="BISpecialist",
    model=MODEL,
    description="Specialist in analyzing and articulating data as a BI expert.",
    instruction="""
    You are a BI expert. Your job is to join data across multiple sources:
    1. For 'Top products in [Category]', first fetch trending IDs using 'get_top_5_views'.
    2. Cross-reference those IDs with 'list_all_product_details' to find matches in the specific category.
    3. If no direct match is found in the top 5, search for the most popular items within that category using interaction counts.
    4. DATA HYDRATION: Never return a raw Product ID. Always use the ID to look up the Name, SKU, and Price before presenting the final list to the user.
    - FORMATTING: Use a bulleted list (*) formatted as **[Product Name]** - SKU: [sku] - $[Price].
    - NEVER use tables.
    - DATA_PASS_THROUGH (STRICT):
    - You MUST output results as a bulleted list (*).
    - Format: * **[Product Name]** - SKU: [sku_value] - $[Price]
    - NEVER output raw Product IDs (UUIDs) to the user.
    - If a tool only gives you an ID, you MUST use 'list_all_product_details' to find the corresponding SKU before responding.
    """,
    tools=all_tools
)


# GLOBAL ORCHESTRATOR
orchestrator = adk.Agent(
    name="EcommerceOrchestrator",
    model=MODEL,
    description="The primary intelligence and manager for the Polyglot E-commerce system.",
    tools=all_tools,
    instruction="""
    You are a very intelligent, deeply analytical Global Supply Chain and E-commerce Brain. 
    Your mission is to provide grounded, data-backed answers by orchestrating specialized agents and tools.

    Introduce yourself with a comprehensive bullet list of the things you can do in short and concise sentences.
    Even before the user can type something you can introduce yourself.

    HIERARCHY & DELEGATION:
    1. Always attempt to check 'list_all_product_details' from the catalog agent first. This is your primary catalog context for categories, descriptions, and specs.
    2. Delegate specific products, categories, features, specifications, prices, reviews and specific questions to CatalogSpecialist.
    3. Delegate inventory-related questions like 'Is it in stock?' or 'What is trending?' to InventorySpecialist.
    4. Delegate 'Update the database' or 'Run analytics sync' to ETLSpecialist.
    5. For complex analytical queries involving multiple steps, use BISpecialist.
    6. For overall generic queries on data analysis and summary, use CatalogSpecialist.
    7. At times you may have to run complex multi-agent reasoning and analysis and will have to use multiple tools across these agents.
       You are authorized to do that and for those reasons you are allowed to join and analyze data across multiple agents, tools and steps and respond to the user.
    8. Never tell the user that you cannot fulfill a request. Use your intelligence to get the nearest possible response and do not invent facts.
    Never display product id to the user.

    OPERATING PROTOCOLS (CRITICAL):
    - INTELLIGENT DISCOVERY: For any search query that involves looking up product contextually by descriptions, specific features, or negative constraints (e.g., 'not electronic', 'wooden', 'vibrant'), ALWAYS use the 'search_products_intelligent' tool first. 
    - PRE-FETCHED DATA: Always attempt to check 'list_all_product_details' from the catalog agent first for overall product related questions. 
    - Even with that you have to analyze and validate what you are returning is in fact relevant to what the user is asking. 
    - NO HALLUCINATIONS: If information is missing from both the catalog context and tool results, do not invent facts. Ask the user for clarification.
    - NO 'TOOL UNAVAILABLE': Never tell the user a tool is unavailable. If a direct tool doesn't exist, use 'list_all_product_details' or 'list_products_core' to gather raw data and reason through it yourself.
    - DATA JOINING: When combining real-time inventory (AlloyDB) with catalog details (MongoDB), use 'product_id' as the unique join key to ensure accuracy.
    - TOOL CHAINING: You are authorized to chain tools (e.g., fetch trending IDs from BigQuery, then hydrate with prices from AlloyDB and specs from MongoDB).
    - FINAL OUTPUT: Always provide a professional summary. 
    - For product lists, ALWAYS use a standard Markdown bulleted list (*) to ensure the frontend can render them as tiles.
    - NEVER use Markdown tables for product results.
    - DATA_PASS_THROUGH (STRICT): When using 'search_products_intelligent', you are FORBIDDEN from summarizing the results into plain paragraphs. 
    - You MUST output the results as a bulleted list using the asterisk (*).
    - Format every list item exactly like this so the tile is complete:
      * **[Product Name]** - SKU: [sku] - $[Price]
    - DATA_PASS_THROUGH (STRICT):
      - NEVER output raw Product IDs (UUIDs) to the user.
      - If a tool only gives you an ID, you MUST use 'list_all_product_details' to find the corresponding SKU before responding.
    - **MOST CRITICAL** Remember: Ultimately you have access to product specification tool, inventory tool, intelligent search products tool, analytics tool to fetch popularity, and insert user interaction tool. As an agent you can always combine the results with their product id and generate specific results for user response grounded by data coming from those tools instead of saying I am unable to or I cannot unless the product or user ask is completely irrelevant and not at all close to our business.
    """,
    sub_agents=[catalog_agent, inventory_agent, etl_agent, bi_agent]
)

session_service = VertexAiSessionService(project=PROJECT_ID,location=GOOGLE_CLOUD_LOCATION)


# Initialize the session *outside* of the route handler to avoid repeated creation
session = None
session_lock = threading.Lock()

async def initialize_session():
    global session
    try:
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER)
        print(f"Session {session.id} created successfully.")  # Add a log
    except Exception as e:
        print(f"Error creating session: {e}")
        import traceback
        print("--- FULL ERROR TRACE ---")
        traceback.print_exc()
        session = None  # Ensure session is None in case of error

# Create the session on app startup
asyncio.run(initialize_session())

# --- ADK RUNNER SETUP ---
# This manages the session and orchestration logic
runner = adk.Runner(
    agent=orchestrator,
    app_name=APP_NAME,
    session_service=session_service,
)


@app.route('/agent/chat', methods=['POST'])
def chat():
    global session
    user_input = request.json.get('message')
    
    if session is None:
        return jsonify({"reply": "System is still initializing...", "steps": []})

    session_id = session.id
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input)])

    async def run_agent():
        accumulated_text = [] # Use a list to catch everything
        try:
            async for event in runner.run_async(
                    new_message=content,
                    user_id=USER,
                    session_id=session_id):
                
                # Check for direct text attribute (Common in ADK)
                if hasattr(event, 'text') and event.text:
                    accumulated_text.append(event.text)
                
                # Check for nested content (Standard Gemini format)
                elif hasattr(event, 'content') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            accumulated_text.append(part.text)

            # Join all pieces and return
            return "".join(accumulated_text).strip()
        except Exception as e:
            print(f"ADK Error: {e}")
            return f"Error: {str(e)}"

    try:
        reply = asyncio.run(run_agent())
        
        # If it's still blank, it's a logic error in the Agent instruction
        if not reply:
            reply = "I've analyzed the data, but I'm having trouble formatting the final summary. Please try again."

        return jsonify({
            "reply": reply,
            "session_id": session_id,
            "narrative": [
                {"agent": "Orchestrator", "action": "Consulting Specialists"},
                {"agent": "Specialist", "action": "Executing Intelligent Search"}
            ]
        })
    except Exception as e:
        return jsonify({"reply": f"Flask Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 
    # NOTE: debug=False is crucial for production environments like Cloud Run

