# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st

from math import sqrt
from PIL import Image
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from utils import *

# Loading Image using PIL
im = Image.open('./src/images/images.png')
# Adding Image to web app
st.set_page_config(page_title="Inventory Management System App", page_icon = im)

# CSS
remote_css("https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.0/semantic.min.css")
local_css("style/style.css")

# Load data
supplier_info, sales_data, inventory_level = load_data()


# Start the Streamlit app
def run():
    st.title('Inventory Management System')
    st.markdown(f"""<br><br>""", unsafe_allow_html=True)
    # Select a product
    st.write("Predict future sales using past data. This helps know how much we might sell in the future.")
    product_option = st.selectbox('Select a product:', list(supplier_info['product'].unique()))
    
    # Load sales data for the selected product
    product_sales = sales_data[sales_data['product'] == product_option]
    product_supplier = supplier_info[supplier_info['product'] == product_option]
    
    # Sales forecast
    fig, future_dates = sales_forecasting(product_sales)
    # Plot sales & forecast Data
    st.plotly_chart(fig)
    st.markdown(f"""<br><br>""", unsafe_allow_html=True)
    # Simulation 
    st.write("### Simulation Inventory Management")
    st.write("Calculates the Economic Order Quantity (EOQ), the number of orders needed for the next month, and the suggested reorder date for each component of a selected car.")
    st.markdown(f"""<br><br>""", unsafe_allow_html=True)

    product_components = inventory_level[inventory_level['product'] == product_option]
    # Initialize the EOQ Results Dataframe
    eoq_results = pd.DataFrame(columns=['Component', 'EOQ', 'Number of Orders for Next 365 Days', 'Reorder Days'])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("")
    with col2:
        st.write("Supplier Lead Time")
    with col3:
        st.write("Supplier Cost per Unit €")
    with col4:
        st.write("Fixed Cost per Order €")
    # Iterate over each component to generate inputs and calculate EOQ
    for component in product_components['component'].unique():
        col1, col2, col3, col4 = st.columns(4)
        default_supplier_lead_time = int(product_supplier[product_supplier['component'] == component]['lead_time_days'].values[0])
        default_supplier_cost_per_unit = float(product_supplier[product_supplier['component'] == component]['cost_per_unit'].values[0])
        default_current_stock = int(product_components[product_components['component'] == component]['inventory_level'].values[0])
        default_K = np.random.randint(25, 80)  # fixed cost per order, you might need to adjust this according to your situation
        with col1:
            st.text(" \n")
            st.text(" \n")
            st.text(" \n")
            st.write(f"##### {component}", )
        with col2:
            supplier_lead_time = st.number_input(label = "", key= "one"+str(component), min_value=1, max_value=30, value=default_supplier_lead_time)
        with col3:        
            supplier_cost_per_unit = st.number_input(f'', key= "two"+str(component), min_value=0.0, max_value=1000.0, value=default_supplier_cost_per_unit, step=0.1)
        with col4:
            K = st.number_input(f'', key= "three"+str(component), min_value=0, max_value=1000, value=default_K, step=10)
        holding_cost = 0.2 * supplier_cost_per_unit  # 20% of the unit cost
        D = sum(future_dates['y_forecast'].values) * int(product_components[product_components['component'] == component]['required_quantity'].values[0]) # Add values required quantity
        h = holding_cost
        L = supplier_lead_time  # Lead time for supplier
        Z = norm.ppf(0.90)  # Z-score for 95% service level, change according to your needs
        std_dev = product_sales['sales'].std()  # Assuming sales is a good proxy for demand. Replace with actual demand standard deviation if available

        eoq, safety_stock = calculate_eoq(D, K, h, L, Z, std_dev)
        num_orders = int(D / eoq)
        
        # Calculate the number of orders for the worst case scenario
        D_worst_case = sum(future_dates['y_forecast-hi-80'].values) * int(product_components[product_components['component'] == component]['required_quantity'].values[0]) # Add values required quantity
        h_worst_case = holding_cost
        L_worst_case = supplier_lead_time  # Lead time for supplier
        Z_worst_case = norm.ppf(0.90)  # Z-score for 95% service level, change according to your needs
        std_dev_worst_case = product_sales['sales'].std()  # Assuming sales is a good proxy for demand. Replace with actual demand standard deviation if available

        eoq_worst_case, safety_stock_worst_case = calculate_eoq(D_worst_case, K, h_worst_case, L_worst_case, Z_worst_case, std_dev_worst_case)
        num_orders_worst_case = int(D_worst_case / eoq_worst_case)
        
        # Calculate the reorder points
        safety_stock = calculate_safety_stock(Z, std_dev, L)  # Set z_value, std_dev, lead_time appropriately
        reorder_point = D * (supplier_lead_time / 30) + safety_stock
        
        # Calculate reorders for the next month
        reorder_days = []
        current_inventory = default_current_stock + eoq  # This is initial inventory level
        cumulative_demand = 0
        lead_time_demand = 0
        
        # Get the required quantity of the component for the product
        required_quantity = int(product_components[product_components['component'] == component]['required_quantity'].values[0])

        for idx, forecast in future_dates.iterrows():
            cumulative_demand += forecast['y_forecast'] * required_quantity  # 'y_forecast' is expected demand
            
            if idx < supplier_lead_time:
                lead_time_demand += forecast['y_forecast'] * required_quantity
            elif idx >= supplier_lead_time:
                lead_time_demand += (forecast['y_forecast'] * required_quantity) - (future_dates.loc[idx - supplier_lead_time, 'y_forecast'] * required_quantity)
            
            if lead_time_demand >= current_inventory:
                reorder_days.append(pd.to_datetime(forecast['date']).to_pydatetime().date().strftime('%Y-%m-%d'))
  # Add the date to reorder_days
                current_inventory += eoq  # Add EOQ to inventory each time we reorder
                # Reset lead_time_demand
                lead_time_demand = 0
            # If we've hit a reorder point, break the loop.
            if len(reorder_days) > 0:
                break
            
        total_inventory_cost = current_inventory * supplier_cost_per_unit
        expected_demand = sum(future_dates['y_forecast'].values[:supplier_lead_time]) * required_quantity
        stockout_risk = max(0, expected_demand - reorder_point) / expected_demand
        expected_supply = current_inventory + eoq
        excess_inventory_risk = round(max(0, expected_supply - expected_demand) / expected_supply, 1)
        # Add EOQ results to the DataFrame
        new_row = pd.DataFrame({'Component': [component],
                        'Current Stock': [default_current_stock],
                        'EOQ': [eoq],
                        'Number of Units for Next Month': [num_orders],
                        'Next Months Projected Unit Usage': [D], 
                        'Reorder Day': [reorder_days[0]] if reorder_days else ['No Need'],
                        'Lead Time (days)': [supplier_lead_time],
                        'Safety Stock Level': [safety_stock],
                        'Total Inventory Cost': [total_inventory_cost],
                        'Stockout Risk': [stockout_risk],
                        'Excess Inventory Risk': [excess_inventory_risk], 
                        'Worst Case Units for Next Month': [num_orders_worst_case]})


        eoq_results = pd.concat([eoq_results, new_row], ignore_index=True)
    # Display EOQ results in a table
    st.markdown(f"""<br><br><br><br>""", unsafe_allow_html=True)
    table_scorecard = """<div id="mydiv" class="ui centered cards">"""

    for index, row in eoq_results.iterrows():
        table_scorecard += f"""
        <div class="card">   
                <div class="content viewbackground">
                        <div class=" header smallheader">{str(row['Component'])}</div>
                </div>
                <div class="content">
                    <div class="description"><br>
                        <div class="column kpi number">{str(int(row['Number of Units for Next Month']))}<br>
                            <p class="kpi text">Number of Units for Next Month</p>
                        </div>
                        <div class="column kpi number">{str(row['Reorder Day'])}<br>
                            <p class="kpi text">Suggested Reorder Date</p>
                        </div>
                    </div>
                </div>
                <div class="extra content">
                    <div class="meta"><i class="warehouse icon"></i> <strong>Current Stock:</strong> {str(int(row['Current Stock'])) + ' units'}</div>
                    <div class="meta"><i class="wait icon"></i> <strong>Lead Time:</strong> {str(int(row['Lead Time (days)'])) + ' days'}</div>
                    <div class="meta"><i class="shield icon"></i> <strong>Safety Stock Level:</strong> {str(round(row['Safety Stock Level'])) + ' units'}</div>
                    <div class="meta"><i class="dollar sign icon"></i> <strong>Total Inventory Cost:</strong> {str(int(row['Total Inventory Cost'])) + ' €'}</div>
                    <div class="meta"><i class="warning sign icon"></i> <strong>Stockout Risk:</strong> {str(int(row['Stockout Risk']))}</div>
                    <div class="meta"><i class="balance scale icon"></i> <strong>Excess Inventory Risk:</strong> {str(float(row['Excess Inventory Risk']))}</div>
                    <div class="meta"><i class="cubes icon"></i> <strong>Worst Case Units for Next Month:</strong> {str(int(row['Worst Case Units for Next Month']))}</div>
                </div>
        </div>"""


    st.markdown(table_scorecard, unsafe_allow_html=True)
    
    st.markdown(f"""<br><br><br><br>""", unsafe_allow_html=True)

    csv = convert_df(eoq_results)

    st.download_button(
        "Download results into CSV",
        csv,
        "Inventory_Management_System_Results.csv",
        "text/csv",
        key='download-csv'
        )
    
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
       
    st.markdown(hide_default_format, unsafe_allow_html=True)

    st.markdown(f"""<br><br><br><br>""", unsafe_allow_html=True)

    st.markdown("""
                Glossary:
            - Lead Time: The time from placing to receiving an order.
            - Safety Stock Level: Extra stock to prevent shortfalls from supply-demand uncertainties.
            - Total Inventory Cost: The cumulative cost of ordering, holding, and storing goods.
            - Stockout Risk: Risk of running out of an item, potentially hurting sales and customer loyalty.
            - Excess Inventory Risk: Risk of having too much stock, leading to higher costs and potential wastage.
            - Worst Case Units for Next Month: Number of units to buy regarding the upper cofidence interval from the forecast model.""")
if __name__ == '__main__':
    run()
