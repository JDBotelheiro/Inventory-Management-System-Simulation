# Inventory_Management_System_Simulation
An interactive Python web application that assists users in inventory management and sales forecasting. This app can help optimize inventory levels and reduce costs related to overstock and stockouts.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jdbotelheiro-inventory-management-system-simulation-app-cns1rf.streamlit.app/)

## Description

The application reads supplier and inventory data, displays a sales forecast for each product, and calculates the Economic Order Quantity (EOQ), safety stock levels, reorder points, and various risks for each component of a selected product. It displays these results on a user-friendly dashboard and provides a download option for the results.

## Key Features

1. **Product Selection:** Users can select a specific product for which they want to manage inventory.
2. **Sales Forecasting:** The app displays a sales forecast for the selected product.
3. **Inventory Simulation:** The app calculates and displays the EOQ, the number of orders needed for the next year, and the suggested reorder date for each component of the selected product.
4. **Inventory Cost and Risk Calculation:** The app calculates and displays various costs and risks associated with inventory management, such as total inventory cost, stockout risk, and excess inventory risk.
5. **Download Results:** Users can download the results in a CSV file.

## Usage

- Import the required libraries and load data.
- Run the app, which will display a title and select product dropdown.
- Select a product from the dropdown menu.
- View the sales forecast and inventory management simulation results for the selected product.
- Optionally, download the results as a CSV file.

## Key Functions

- `run()`: Main function that runs the Streamlit app.
- `calculate_eoq()`: Helper function to calculate the Economic Order Quantity and safety stock.
- `calculate_safety_stock()`: Helper function to calculate the safety stock level.

## Dependencies

- pandas
- numpy
- streamlit
- math
- PIL
- scipy.stats
- sklearn.model_selection
- utils
