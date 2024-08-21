from supabase import create_client, Client
import streamlit as st
import pandas as pd

# Supabase connection
url:str= st.secrets['supabase_url']  # 你的 Supabase URL
key :str = st.secrets['supabase_key']  # 你的 Supabase API 密钥
supabase: Client = create_client(url, key)

# Fetch data
def view_all_data():
    response = supabase.table("testing_data").select("*").execute()
    data = response.data
    data = pd.DataFrame(data)

    return data




if __name__ == "__main__":
    df=view_all_data()
    print(df.columns)