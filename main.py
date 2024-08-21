import streamlit as st
import plotly.express as px
import pandas as pd
from supabase_conn import view_all_data
import numpy as np
from joblib import load

theme_plotly = None  # or you can use streamlit theme

# create page config
st.set_page_config("Business Analytics Dashboard", page_icon="", layout="wide")
st.subheader("Data Center")

# HTML and JavaScript code for the clock
html_code = """
<!DOCTYPE html>
<html>
<head>
    <style>
        .clock {
            font-size: 50px;
            font-weight: bold;
            color: white; /* 设置时间的颜色为白色 */
        }
        .clock-text {
            font-size: 25px;
            color: white; /* 设置文字的颜色为白色 */
        }
    </style>
    <script>
        function startTime() {
            const today = new Date();
            let h = today.getHours();
            let m = today.getMinutes();
            let s = today.getSeconds();
            m = checkTime(m);
            s = checkTime(s);
            document.getElementById('time').innerHTML = h + ":" + m + ":" + s;
            setTimeout(startTime, 1000);
        }

        function checkTime(i) {
            if (i < 10) {i = "0" + i};  // 在数字小于10时，在数字前加一个"0"
            return i;
        }
    </script>
</head>
<body onload="startTime()">
    <div class="clock-text">Now <strong>Shanghai Time</strong></div>
    <div id="time" class="clock"></div>
</body>
</html>
"""

st.components.v1.html(html_code, height=100)

# Load CSS styles
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Process data from query
result = view_all_data()
df = pd.DataFrame(result, columns=['bet_day', 'api_name', 'order_type', 'name', 
                                   'order_number', 'order_amount', 'net_amount', 'valid_order_amount'])

# Sidebar filters
st.sidebar.header("Filter Dataset")
order_type = st.sidebar.multiselect(
    label="Order Type Filter",
    options=df["order_type"].unique(),
    default=df["order_type"].unique(),
)

st.sidebar.header("Filter Api Name")
api_name = st.sidebar.multiselect(
    label="API Name Filter",
    options=df["api_name"].unique(),
    default=df["api_name"].unique(),
)

# Process query
df_selection = df.query(
    "order_type == @order_type & api_name == @api_name"
)

def metrics():
    from streamlit_extras.metric_cards import style_metric_cards
    col1, col2, col3, col4 = st.columns(4)

    # active users
    col1.metric("Active User", value=df_selection['name'].nunique(), delta="active users")
    # order numbers
    col2.metric("Order Number", value=f"{df_selection['order_number'].sum():,.0f}", delta='Order Number')
    # order amount
    col3.metric("Valid Order Amount", value=f"{df_selection['valid_order_amount'].sum():,.0f}", delta='Valid Order Amount')
    # Net amount
    col4.metric("Net Amount", value=f"{df_selection['net_amount'].sum():,.0f}", delta='Net Amount')

    style_metric_cards(background_color="#071021", border_color="#1f66bd")

# pie chart
div1, div2 = st.columns(2)
def pie():
    with div1:
        fig = px.pie(df_selection, values="order_number", names="order_type", title="Order Number by Order Type")
        fig.update_layout(legend_title="Order Type", legend_y=0.9)
        fig.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

# bar chart
def bar():
    with div2:
        fig = px.bar(df_selection, y="valid_order_amount", x="order_type", text_auto='.2s', title="Valid Order Amount by Order Type")
        fig.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

# Data Table
def table():
    with st.expander("My Database Table"):
       shwdata = st.multiselect("Filter Dataset", df_selection.columns, default=['bet_day', 'api_name', 'order_type', 'name', 
                                                                                 'order_number', 'order_amount', 'net_amount', 'valid_order_amount'])
       st.dataframe(df_selection[shwdata], use_container_width=True)

# Load Machine Learning Model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    pipe = load('model/model.joblib')
    return pipe

def make_prediction(pipe):
    miles = st.session_state["miles"]
    year = st.session_state["year"]
    make = st.session_state["make"]
    model = st.session_state["model"]
    engine_size = st.session_state["engine_size"]
    province = st.session_state["province"]

    X_pred = np.array([miles, year, make, model, engine_size, province]).reshape(1,-1)

    pred = pipe.predict(X_pred)
    pred = round(pred[0], 2)

    st.session_state["pred"] = pred

# side Navigation
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(    
        menu_title="Main Menu",
        options=["Home", "Table", "Betdata", "Car Price Prediction"],
        icons=["house", "table", "graph-up-arrow", "car-front"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

if selected == "Home":
    pie()
    bar()
    metrics()

elif selected == "Table":
    table()
    st.dataframe(df_selection.describe().T, use_container_width=True)

elif selected == "Car Price Prediction":
    # Initialize session state variables
    if "pred" not in st.session_state:
        st.session_state["pred"] = None

    if "miles" not in st.session_state:
        st.session_state["miles"] = 86132.0

    if "year" not in st.session_state:
        st.session_state["year"] = 2001

    if "make" not in st.session_state:
        st.session_state["make"] = 'toyota'

    if "model" not in st.session_state:
        st.session_state["model"] = 'Prius'

    if "engine_size" not in st.session_state:
        st.session_state["engine_size"] = 1.5

    if "province" not in st.session_state:
        st.session_state["province"] = 'NB'

    st.title("🍁Used car price calculator")

    pipe = load_model()

    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input("Miles", value=86132.0, min_value=0.0, step=0.1, key="miles")
            st.selectbox("Model", index=0, key="model", options=['Prius', 'Highlander', 'Civic', 'Accord', 'Corolla', 'Ridgeline',
               'Odyssey', 'CR-V', 'Pilot', 'Camry Solara', 'Matrix', 'RAV4',
               'Rav4', 'HR-V', 'Fit', 'Yaris', 'Yaris iA', 'Tacoma', 'Camry',
               'Avalon', 'Venza', 'Sienna', 'Passport', 'Accord Crosstour',
               'Crosstour', 'Element', 'Tundra', 'Sequoia', 'Corolla Hatchback',
               '4Runner', 'Echo', 'Tercel', 'MR2 Spyder', 'FJ Cruiser',
               'Corolla iM', 'C-HR', 'Civic Hatchback', '86', 'S2000', 'Supra',
               'Insight', 'Clarity', 'CR-Z', 'Prius Prime', 'Prius Plug-In',
               'Prius c', 'Prius C', 'Prius v'])
        with col2:
            st.number_input("Year", value=2001, min_value=1886, step=1, key="year")
            st.number_input("Engine size (L)", value=1.5, key="engine_size", min_value=0.9, step=0.1)
        with col3:
            st.selectbox("Make", key="make", index=0, options=['toyota', 'honda'])
            st.selectbox("Province", index=0, key="province", options=['NB', 'QC', 'BC', 'ON', 'AB', 'MB', 'SK', 'NS', 'PE', 'NL', 'YT', 'NC', 'OH','SC'])
        
        st.form_submit_button("Calculate", type="primary", on_click=make_prediction, kwargs=dict(pipe=pipe))

    if st.session_state["pred"] is not None:
        st.subheader(f"The estimated car price is {st.session_state.pred}$")
    else:
        st.write("Input information and click on Calculate to get an estimated price")
    
    st.write(st.session_state)
