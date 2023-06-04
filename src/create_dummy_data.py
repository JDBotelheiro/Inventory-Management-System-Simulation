import pandas as pd
import numpy as np
import datetime
import random

# Set a seed for reproducibility
np.random.seed(0)

# Define a list of products
products = ['Tesla S', 'Tesla Y', 'Tesla 3']

# Define a list of components required for each product
components = ['Aluminum', 'High-strength steel', 'Glass', 'Carbon fiber', 'Lithium battery']

# Define a dictionary to hold the quantity of each component required for each product
product_components = {
    'Tesla S': {'Aluminum': 5, 'High-strength steel': 2, 'Glass': 5, 'Carbon fiber': 4, 'Lithium battery': 1},
    'Tesla Y': {'Aluminum': 5, 'High-strength steel': 2, 'Glass': 5, 'Carbon fiber': 4, 'Lithium battery': 1},
    'Tesla 3': {'Aluminum': 5, 'High-strength steel': 2, 'Glass': 5, 'Carbon fiber': 4, 'Lithium battery': 1}

}

# Define empty dataframes for sales data, inventory levels, and supplier info
sales_data = pd.DataFrame()
inventory_level = pd.DataFrame()
supplier_info = pd.DataFrame()

# Generate data for each product
for product in products:
    # Historical Sales Data
    # We'll assume we have data for the past three years (1095 days)
    dates = pd.date_range(end = datetime.datetime.today(), periods = 1095).tolist()
    sales = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(50, 200, size=1095), # random sales data between 50 and 200 units
        'product': product
    })
    # Add some fluctuation in sales to simulate promotions or out-of-stock situations
    sales['sales'] = sales['sales'] + np.random.randint(-15, 15, size=1095)
    
    # Append to the sales_data dataframe
    sales_data = sales_data.append(sales, ignore_index=True)

    # Current Inventory Level
    for component in components:
        # We'll assume we currently have 5000 units in stock for each component
        inventory = pd.DataFrame({
            'product': [product],
            'component': [component],
            'inventory_level': [random.randint(100, 400)],
            'required_quantity': [product_components[product][component]]
        })
        # Append to the inventory_level dataframe
        inventory_level = inventory_level.append(inventory, ignore_index=True)

        # Supplier Information
        # We'll assume it takes 14 days for an order to arrive and each unit costs $10
        supplier = pd.DataFrame({
            'product': [product],
            'component': [component],
            'lead_time_days': [random.randint(2, 20)],
            'cost_per_unit': [random.randint(15, 200)]
        })
        # Append to the supplier_info dataframe
        supplier_info = supplier_info.append(supplier, ignore_index=True)

# Seasonality/Trends
# We'll create a simple trend where sales increase by 10% every quarter and simulate seasonality by boosting sales in Q4 (holiday season)
quarterly_growth_rate = 0.10
sales_data['quarter'] = sales_data['date'].dt.quarter
sales_data['sales'] = sales_data['sales'] * (1 + sales_data['quarter'] * quarterly_growth_rate)
sales_data.loc[sales_data['quarter'] == 4, 'sales'] = sales_data.loc[sales_data['quarter'] == 4, 'sales'] * 1.2
print(sales_data.head())
# Save the dataframes to Parquet files
sales_data.to_parquet('./data/sales_data.parquet')
inventory_level.to_parquet('./data/inventory_level.parquet')
supplier_info.to_parquet('./data/supplier_info.parquet')
