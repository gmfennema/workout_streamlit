import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

def fetch_events():
    """Fetch all events from the SQLite database."""
    conn = sqlite3.connect('calendar_events.db')
    c = conn.cursor()
    c.execute('SELECT * FROM events ORDER BY Start_Datetime DESC')
    events = c.fetchall()
    conn.close()
    return events

def count_coffee_emojis(title):
    """Count the number of coffee emojis in a title."""
    return title.count('☕')

def count_beer_emojis(title):
    """Count the number of beer emojis in a title."""
    return title.count('🍺')

def count_kiss_emojis(title):
    """Count the number of kiss emojis in a title."""
    return title.count('😘')

def get_category(color_id):
    """Map color ID to category."""
    category_map = {
        0: 'Family Time',
        1: 'Social',
        2: 'Navigators',
        3: 'Chores',
        4: 'Ignore',
        5: 'Spiritual',
        6: 'N/A',
        7: 'Consulting',
        8: 'Personal',
        9: 'Exercise'
    }
    try:
        # Explicitly convert to integer
        color_id_int = int(color_id)
        return category_map.get(color_id_int, f'Unknown: {color_id}')
    except (ValueError, TypeError):
        return f'Invalid Color ID: {color_id}'

def home_page():
    st.title("Calendar Events")

    events = fetch_events()

    if events:
        # Create a DataFrame from the events, including the event_id
        df = pd.DataFrame(events, columns=["Event ID", "Start", "End", "Title", "Color ID"])
        
        # Add coffee and beer count columns
        df['Coffee_Count'] = df['Title'].apply(count_coffee_emojis)
        df['Beer_Count'] = df['Title'].apply(count_beer_emojis)
        
        # Add category column
        df['Category'] = df['Color ID'].apply(get_category)
        
        st.dataframe(df)  # Display the DataFrame in a table format
    else:
        st.write("No events found.")

    # Display total number of events
    total_events = len(events)
    st.sidebar.write(f"Total Events: {total_events}")

def display_year(z, year: int = None, month_lines: bool = True, fig=None, row: int = None):
    if year is None:
        year = datetime.now().year
        
    d1 = datetime(year, 1, 1)
    d2 = datetime(year, 12, 31)

    number_of_days = (d2 - d1).days + 1
    
    # Adjust z length if it's a partial year
    if len(z) < number_of_days:
        z = np.concatenate([z, np.zeros(number_of_days - len(z))])
    elif len(z) > number_of_days:
        z = z[:number_of_days]
    
    data = z

    delta = d2 - d1

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days =   [31,    28,    31,     30,    31,     30, 
                   31,    31,    30,    31,    30,    31]
    if number_of_days == 366:  # leap year
        month_days[1] = 29
    month_positions = (np.cumsum(month_days) - 15)/7

    dates_in_year = [d1 + timedelta(days=i) for i in range(delta.days + 1)]
    weekdays_in_year = [date.weekday() for date in dates_in_year]  # 0=Monday

    weeknumber_of_dates = [date.isocalendar().week for date in dates_in_year]
    # Handle edge cases where ISO week might be 53 or 0
    weeknumber_of_dates = [w if w <= 53 else 53 for w in weeknumber_of_dates]

    text = [str(date.date()) for date in dates_in_year]  # Hover text

    # GitHub-like colors: light gray to dark blue
    colorscale = [
        [0.0, '#ebedf0'],
        [1.0, '#02A6F4']
    ]
    
    heatmap = go.Heatmap(
        x=weeknumber_of_dates,
        y=weekdays_in_year,
        z=data,
        text=text,
        hoverinfo='text',
        xgap=3,  # Gap between cells
        ygap=3,  # Gap between cells
        showscale=False,
        colorscale=colorscale,
        zmin=0,
        zmax=5  # Adjust based on your data's max count
    )
    
    data_traces = [heatmap]
        
    if month_lines:
        kwargs = dict(
            mode='lines',
            line=dict(
                color='#9e9e9e',
                width=1,
            ),
            hoverinfo='skip',
        )
        
        for date, dow, wkn in zip(dates_in_year, weekdays_in_year, weeknumber_of_dates):
            if date.day == 1:
                data_traces += [
                    go.Scatter(
                        x=[wkn - 0.5, wkn - 0.5],
                        y=[dow - 0.5, 6.5],
                        **kwargs,
                    )
                ]
                if dow:
                    data_traces += [
                        go.Scatter(
                            x=[wkn - 0.5, wkn + 0.5],
                            y=[dow - 0.5, dow - 0.5],
                            **kwargs,
                        ),
                        go.Scatter(
                            x=[wkn + 0.5, wkn + 0.5],
                            y=[dow - 0.5, -0.5],
                            **kwargs,
                        )
                    ]
                    
    layout = go.Layout(
        height=250,
        yaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            autorange="reversed",
        ),
        xaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=month_names,
            tickvals=month_positions,
        ),
        font={'size':10, 'color':'#9e9e9e'},
        plot_bgcolor=('#fff'),
        margin = dict(t=40, l=20, r=20, b=20),
        showlegend=False,
    )

    if fig is None:
        fig = go.Figure(data=data_traces, layout=layout)
    else:
        fig.add_traces(data_traces, rows=[(row+1)]*len(data_traces), cols=[1]*len(data_traces))
        fig.update_layout(layout)
        fig.update_xaxes(layout['xaxis'])
        fig.update_yaxes(layout['yaxis'])

    return fig

def display_year(z, year: int = None, month_lines: bool = True, fig=None, row: int = None):
    if year is None:
        year = datetime.now().year
        
    d1 = datetime(year, 1, 1)
    d2 = datetime(year, 12, 31)

    number_of_days = (d2 - d1).days + 1
    
    data = z  # z is already padded to the full year length

    delta = d2 - d1

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days =   [31,    28,    31,     30,    31,     30, 
                   31,    31,    30,    31,    30,    31]
    if number_of_days == 366:  # leap year
        month_days[1] = 29
    month_positions = (np.cumsum(month_days) - 15)/7

    dates_in_year = [d1 + timedelta(days=i) for i in range(delta.days + 1)]
    weekdays_in_year = [date.weekday() for date in dates_in_year]  # 0=Monday

    weeknumber_of_dates = [date.isocalendar().week for date in dates_in_year]
    # Handle edge cases where ISO week might be 53 or 0
    weeknumber_of_dates = [w if w <= 53 else 53 for w in weeknumber_of_dates]

    text = [str(date.date()) for date in dates_in_year]  # Hover text

    # GitHub-like colors: light gray to dark blue
    colorscale = [
        [0.0, '#ebedf0'],
        [1.0, '#02A6F4']
    ]
    
    heatmap = go.Heatmap(
        x=weeknumber_of_dates,
        y=weekdays_in_year,
        z=data,
        text=text,
        hoverinfo='text',
        xgap=3,  # Gap between cells
        ygap=3,  # Gap between cells
        showscale=False,
        colorscale=colorscale,
        zmin=0,
        zmax=max(data) if max(data) > 0 else 1  # Dynamic zmax based on data
    )
    
    data_traces = [heatmap]
        
    if month_lines:
        kwargs = dict(
            mode='lines',
            line=dict(
                color='#9e9e9e',
                width=1,
            ),
            hoverinfo='skip',
        )
        
        for date, dow, wkn in zip(dates_in_year, weekdays_in_year, weeknumber_of_dates):
            if date.day == 1:
                data_traces += [
                    go.Scatter(
                        x=[wkn - 0.5, wkn - 0.5],
                        y=[dow - 0.5, 6.5],
                        **kwargs,
                    )
                ]
                if dow:
                    data_traces += [
                        go.Scatter(
                            x=[wkn - 0.5, wkn + 0.5],
                            y=[dow - 0.5, dow - 0.5],
                            **kwargs,
                        ),
                        go.Scatter(
                            x=[wkn + 0.5, wkn + 0.5],
                            y=[dow - 0.5, -0.5],
                            **kwargs,
                        )
                    ]
                    
    layout = go.Layout(
        height=250,
        yaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            autorange="reversed",
        ),
        xaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=month_names,
            tickvals=month_positions,
        ),
        font={'size':10, 'color':'#9e9e9e'},
        plot_bgcolor=('#fff'),
        margin = dict(t=40, l=20, r=20, b=20),
        showlegend=False,
    )

    if fig is None:
        fig = go.Figure(data=data_traces, layout=layout)
    else:
        fig.add_traces(data_traces, rows=[(row+1)]*len(data_traces), cols=[1]*len(data_traces))
        fig.update_layout(layout)
        fig.update_xaxes(layout['xaxis'])
        fig.update_yaxes(layout['yaxis'])

    return fig

def display_years(z_yearly, years):
    fig = make_subplots(rows=len(years), cols=1, subplot_titles=[f"Year {year}" for year in years], vertical_spacing=0.05)
    for i, (z, year) in enumerate(zip(z_yearly, years)):
        fig = display_year(z, year=year, fig=fig, row=i)
    fig.update_layout(height=250*len(years))
    return fig

def exercise_page():
    st.title("Exercise")

    # Fetch events and create DataFrame
    events = fetch_events()
    df = pd.DataFrame(events, columns=["Event ID", "Start", "End", "Title", "Color ID"])
    df['Start'] = pd.to_datetime(df['Start'], utc=True)
    df['End'] = pd.to_datetime(df['End'], utc=True)
    df['Category'] = df['Color ID'].apply(get_category)
    
    # Filter for Exercise events and convert dates
    exercise_df = df[df['Category'] == 'Exercise'].copy()
    exercise_df['Start'] = pd.to_datetime(exercise_df['Start'])
    exercise_df['End'] = pd.to_datetime(exercise_df['End'])
    
    # Calculate metrics
    total_distinct_exercise_days = exercise_df['Start'].dt.date.nunique()
    
    # Calculate average times per week
    avg_times_per_week = (total_distinct_exercise_days / pd.to_datetime(df['Start']).dt.date.nunique()) * 7 if pd.to_datetime(df['Start']).dt.date.nunique() > 0 else 0

    # Calculate average duration
    exercise_df['Duration'] = (exercise_df['End'] - exercise_df['Start']).dt.total_seconds() / 3600  # Convert to hours
    avg_duration = exercise_df['Duration'].mean() if not exercise_df.empty else 0

    # Calculate lifting percentage
    lifting_exercises = exercise_df[exercise_df['Title'].str.contains('Lift', case=False, na=False)]
    lifting_percentage = (len(lifting_exercises) / len(exercise_df)) * 100 if len(exercise_df) > 0 else 0

    # Calculate social exercise percentage
    social_exercises = exercise_df[exercise_df['Title'].str.contains('\|', na=False)]
    social_exercise_percentage = (len(social_exercises) / len(exercise_df)) * 100 if len(exercise_df) > 0 else 0

    # Display metrics
    st.subheader("Overview")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 20px; width: 100%;">
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(20% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Total Exercise Days</h5>
            <div style="font-size: 34px; font-weight: bold; color: #02A6F4; align-self: center;">{total_distinct_exercise_days}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(20% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Avg Times per Week</h5>
            <div style="font-size: 34px; font-weight: bold; color: #02A6F4; align-self: center;">{avg_times_per_week:.1f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(20% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Avg Duration (hrs)</h5>
            <div style="font-size: 34px; font-weight: bold; color: #02A6F4; align-self: center;">{avg_duration:.1f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(20% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Lifting %</h5>
            <div style="font-size: 34px; font-weight: bold; color: #02A6F4; align-self: center;">{lifting_percentage:.1f}%</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(20% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Social Exercise %</h5>
            <div style="font-size: 34px; font-weight: bold; color: #02A6F4; align-self: center;">{social_exercise_percentage:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("Exercise Activity")
    if not exercise_df.empty:
        # Define the years to display
        years_to_display = [2024, 2025]
        
        # Prepare z_yearly
        z_yearly = []
        
        for target_year in years_to_display:
            start_date = datetime(target_year, 1, 1).date()
            end_date = datetime.now().date() if target_year == datetime.now().year else datetime(target_year, 12, 31).date()
            
            # Create date range for the year
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Count exercises per day
            daily_counts = exercise_df[
                (exercise_df['Start'].dt.date >= start_date) & 
                (exercise_df['Start'].dt.date <= end_date)
            ]['Start'].dt.date.value_counts().sort_index()
            
            # Reindex to include all dates in the range, fill missing with 0
            daily_counts = daily_counts.reindex(date_range.date, fill_value=0)
            
            # Create z array
            z = daily_counts.values
            
            # Pad z to the full year length (365 or 366 days)
            full_year_days = (datetime(target_year, 12, 31) - datetime(target_year, 1, 1)).days + 1
            if len(z) < full_year_days:
                z = np.concatenate([z, np.zeros(full_year_days - len(z))])
            elif len(z) > full_year_days:
                z = z[:full_year_days]
            
            z_yearly.append(z)
        
        fig = display_years(z_yearly, years_to_display)
        st.plotly_chart(fig, use_container_width=True)

        # Show the raw data in an expandable section
        with st.expander("View Raw Data"):
            st.dataframe(exercise_df.sort_values('Start', ascending=False))
    else:
        st.write("No exercise events found.")

def alcohol_page():
    st.title("Alcohol Consumption")

    # Fetch events and create DataFrame
    events = fetch_events()
    df = pd.DataFrame(events, columns=["Event ID", "Start", "End", "Title", "Color ID"])
    
    # Convert Start and End to datetime with UTC timezone
    df['Start'] = pd.to_datetime(df['Start'], utc=True)
    df['End'] = pd.to_datetime(df['End'], utc=True)
    
    # Filter for Alcohol events (events with beer emoji)
    alcohol_df = df[df['Title'].str.contains('🍺', na=False)].copy()
    
    # Count drinks per event (number of beer emojis)
    alcohol_df['Drink_Count'] = alcohol_df['Title'].apply(count_beer_emojis)
    
    # Calculate metrics
    # 1. Avg Drinks per Week
    total_days = (df['Start'].max() - df['Start'].min()).days + 1
    total_weeks = total_days / 7
    total_drinks = alcohol_df['Drink_Count'].sum()
    avg_drinks_per_week = total_drinks / total_weeks
    
    # 2. Drinks in Last 30 Days
    # Use UTC timezone for thirty_days_ago
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    drinks_last_30_days = alcohol_df[alcohol_df['Start'] >= thirty_days_ago]['Drink_Count'].sum()
    
    # 3. Avg Drinks at a Time
    drinking_days = alcohol_df.groupby(alcohol_df['Start'].dt.date)['Drink_Count'].sum()
    avg_drinks_per_drinking_day = drinking_days.mean() if not drinking_days.empty else 0
    
    # 4. Day of Week Most Likely to Drink
    alcohol_df['Day_of_Week'] = alcohol_df['Start'].dt.day_name()
    day_of_week_counts = alcohol_df['Day_of_Week'].value_counts()
    most_likely_day = day_of_week_counts.index[0] if not day_of_week_counts.empty else "N/A"
    
    # Display metrics
    st.subheader("Alcohol Consumption Overview")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 20px; width: 100%;">
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Avg Drinks per Week</h5>
            <div style="font-size: 34px; font-weight: bold; color: #A684FF; align-self: center;">{avg_drinks_per_week:.1f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Drinks in Last 30 Days</h5>
            <div style="font-size: 34px; font-weight: bold; color: #A684FF; align-self: center;">{drinks_last_30_days:.0f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Avg Drinks per Drinking Day</h5>
            <div style="font-size: 34px; font-weight: bold; color: #A684FF; align-self: center;">{avg_drinks_per_drinking_day:.1f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Most Likely Drinking Day</h5>
            <div style="font-size: 34px; font-weight: bold; color: #A684FF; align-self: center;">{most_likely_day}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("Alcohol Activity")
    
    if not alcohol_df.empty:
        # Define the years to display
        years_to_display = [2024, 2025]
        
        # Prepare z_yearly
        z_yearly = []
        
        for target_year in years_to_display:
            start_date = datetime(target_year, 1, 1).date()
            end_date = datetime.now().date() if target_year == datetime.now().year else datetime(target_year, 12, 31).date()
            
            # Create date range for the year
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Count drinks per day
            daily_counts = alcohol_df[
                (alcohol_df['Start'].dt.date >= start_date) & 
                (alcohol_df['Start'].dt.date <= end_date)
            ].groupby(alcohol_df['Start'].dt.date)['Drink_Count'].sum().sort_index()
            
            # Reindex to include all dates in the range, fill missing with 0
            daily_counts = daily_counts.reindex(date_range.date, fill_value=0)
            
            # Create z array
            z = daily_counts.values
            
            # Pad z to the full year length (365 or 366 days)
            full_year_days = (datetime(target_year, 12, 31) - datetime(target_year, 1, 1)).days + 1
            if len(z) < full_year_days:
                z = np.concatenate([z, np.zeros(full_year_days - len(z))])
            elif len(z) > full_year_days:
                z = z[:full_year_days]
            
            z_yearly.append(z)
        
        # Modify display_year to use #A684FF color
        def custom_display_year(z, year: int = None, month_lines: bool = True, fig=None, row: int = None):
            colorscale = [
                [0.0, '#ebedf0'],
                [1.0, '#A684FF']
            ]
            
            # Rest of the function remains the same as display_year
            if year is None:
                year = datetime.now().year
                
            d1 = datetime(year, 1, 1)
            d2 = datetime(year, 12, 31)

            number_of_days = (d2 - d1).days + 1
            
            data = z  # z is already padded to the full year length

            delta = d2 - d1

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_days =   [31,    28,    31,     30,    31,     30, 
                           31,    31,    30,    31,    30,    31]
            if number_of_days == 366:  # leap year
                month_days[1] = 29
            month_positions = (np.cumsum(month_days) - 15)/7

            dates_in_year = [d1 + timedelta(days=i) for i in range(delta.days + 1)]
            weekdays_in_year = [date.weekday() for date in dates_in_year]  # 0=Monday

            weeknumber_of_dates = [date.isocalendar().week for date in dates_in_year]
            # Handle edge cases where ISO week might be 53 or 0
            weeknumber_of_dates = [w if w <= 53 else 53 for w in weeknumber_of_dates]

            text = [str(date.date()) for date in dates_in_year]  # Hover text
            
            heatmap = go.Heatmap(
                x=weeknumber_of_dates,
                y=weekdays_in_year,
                z=data,
                text=text,
                hoverinfo='text',
                xgap=3,  # Gap between cells
                ygap=3,  # Gap between cells
                showscale=False,
                colorscale=colorscale,
                zmin=0,
                zmax=max(data) if max(data) > 0 else 1  # Dynamic zmax based on data
            )
            
            data_traces = [heatmap]
                
            if month_lines:
                kwargs = dict(
                    mode='lines',
                    line=dict(
                        color='#9e9e9e',
                        width=1,
                    ),
                    hoverinfo='skip',
                )
                
                for date, dow, wkn in zip(dates_in_year, weekdays_in_year, weeknumber_of_dates):
                    if date.day == 1:
                        data_traces += [
                            go.Scatter(
                                x=[wkn - 0.5, wkn - 0.5],
                                y=[dow - 0.5, 6.5],
                                **kwargs,
                            )
                        ]
                        if dow:
                            data_traces += [
                                go.Scatter(
                                    x=[wkn - 0.5, wkn + 0.5],
                                    y=[dow - 0.5, dow - 0.5],
                                    **kwargs,
                                ),
                                go.Scatter(
                                    x=[wkn + 0.5, wkn + 0.5],
                                    y=[dow - 0.5, -0.5],
                                    **kwargs,
                                )
                            ]
                            
            layout = go.Layout(
                height=250,
                yaxis=dict(
                    showline=False, showgrid=False, zeroline=False,
                    tickmode='array',
                    ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    tickvals=[0, 1, 2, 3, 4, 5, 6],
                    autorange="reversed",
                ),
                xaxis=dict(
                    showline=False, showgrid=False, zeroline=False,
                    tickmode='array',
                    ticktext=month_names,
                    tickvals=month_positions,
                ),
                font={'size':10, 'color':'#9e9e9e'},
                plot_bgcolor=('#fff'),
                margin = dict(t=40, l=20, r=20, b=20),
                showlegend=False,
            )

            if fig is None:
                fig = go.Figure(data=data_traces, layout=layout)
            else:
                fig.add_traces(data_traces, rows=[(row+1)]*len(data_traces), cols=[1]*len(data_traces))
                fig.update_layout(layout)
                fig.update_xaxes(layout['xaxis'])
                fig.update_yaxes(layout['yaxis'])

            return fig
        
        # Use the custom display_years function with the custom color
        def custom_display_years(z_yearly, years):
            fig = make_subplots(
                rows=len(years), 
                cols=1, 
                subplot_titles=[f"Year {year}" for year in years], 
                vertical_spacing=0.05,
                # Set a reasonable aspect ratio that maintains square cells
                specs=[[{"type": "xy"}] for _ in years]
            )
            
            for i, (z, year) in enumerate(zip(z_yearly, years)):
                fig = custom_display_year(z, year=year, fig=fig, row=i)
            
            # Update layout with dynamic height calculation
            # Base height of 150px per year, with some padding
            fig.update_layout(
                height=max(150 * len(years), 250),  # Minimum height of 250px
                # Force subplot y-axes to have 1:1 aspect ratio with x-axes
                yaxis=dict(scaleanchor="x", scaleratio=1),
                yaxis2=dict(scaleanchor="x2", scaleratio=1)
            )
            
            return fig
        
        fig = custom_display_years(z_yearly, years_to_display)
        st.plotly_chart(fig, use_container_width=True)

        # Show the raw data in an expandable section
        with st.expander("View Raw Data"):
            st.dataframe(alcohol_df.sort_values('Start', ascending=False))
    else:
        st.write("No alcohol events found.")

def sex_page():
    st.title("Intimacy")

    # Fetch events and create DataFrame
    events = fetch_events()
    df = pd.DataFrame(events, columns=["Event ID", "Start", "End", "Title", "Color ID"])
    
    # Convert Start and End to datetime with UTC timezone
    df['Start'] = pd.to_datetime(df['Start'], utc=True)
    df['End'] = pd.to_datetime(df['End'], utc=True)
    
    # Filter for Intimacy events (events with kiss emoji)
    sex_df = df[df['Title'].str.contains('😘', na=False)].copy()
    
    # Count encounters per event (number of kiss emojis)
    sex_df['Encounter_Count'] = sex_df['Title'].apply(count_kiss_emojis)
    
    # Calculate metrics
    # 1. Avg Days per Week
    total_days = (df['Start'].max() - df['Start'].min()).days + 1
    total_weeks = total_days / 7
    total_encounters = sex_df['Encounter_Count'].sum()
    avg_days_per_week = total_encounters / total_weeks
    
    # 2. Previous 30 Days (Count total)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    encounters_last_30_days = sex_df[sex_df['Start'] >= thirty_days_ago]['Encounter_Count'].sum()
    
    # 3. Previous 7 Days (Count total)
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    encounters_last_7_days = sex_df[sex_df['Start'] >= seven_days_ago]['Encounter_Count'].sum()
    
    # 4. Day of Week Most Likely
    sex_df['Day_of_Week'] = sex_df['Start'].dt.day_name()
    day_of_week_counts = sex_df['Day_of_Week'].value_counts()
    most_likely_day = day_of_week_counts.index[0] if not day_of_week_counts.empty else "N/A"
    
    # Display metrics
    st.subheader("Intimacy Overview")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 20px; width: 100%;">
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Avg Days per Week</h5>
            <div style="font-size: 34px; font-weight: bold; color: #F6339A; align-self: center;">{avg_days_per_week:.1f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Encounters in Last 30 Days</h5>
            <div style="font-size: 34px; font-weight: bold; color: #F6339A; align-self: center;">{encounters_last_30_days:.0f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Encounters in Last 7 Days</h5>
            <div style="font-size: 34px; font-weight: bold; color: #F6339A; align-self: center;">{encounters_last_7_days:.0f}</div>
        </div>
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; flex-grow: 1; margin: 0 10px; display: flex; flex-direction: column; justify-content: center; width: calc(25% - 20px);">
            <h5 style="margin-bottom: 10px; color: #333;">Most Likely Day</h5>
            <div style="font-size: 34px; font-weight: bold; color: #F6339A; align-self: center;">{most_likely_day}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("Intimacy Activity")
    
    if not sex_df.empty:
        # Define the years to display
        years_to_display = [2024, 2025]
        
        # Prepare z_yearly
        z_yearly = []
        
        for target_year in years_to_display:
            start_date = datetime(target_year, 1, 1).date()
            end_date = datetime.now().date() if target_year == datetime.now().year else datetime(target_year, 12, 31).date()
            
            # Create date range for the year
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Count encounters per day
            daily_counts = sex_df[
                (sex_df['Start'].dt.date >= start_date) & 
                (sex_df['Start'].dt.date <= end_date)
            ].groupby(sex_df['Start'].dt.date)['Encounter_Count'].sum().sort_index()
            
            # Reindex to include all dates in the range, fill missing with 0
            daily_counts = daily_counts.reindex(date_range.date, fill_value=0)
            
            # Create z array
            z = daily_counts.values
            
            # Pad z to the full year length (365 or 366 days)
            full_year_days = (datetime(target_year, 12, 31) - datetime(target_year, 1, 1)).days + 1
            if len(z) < full_year_days:
                z = np.concatenate([z, np.zeros(full_year_days - len(z))])
            elif len(z) > full_year_days:
                z = z[:full_year_days]
            
            z_yearly.append(z)
        
        # Modify display_year to use #F6339A color
        def custom_display_year(z, year: int = None, month_lines: bool = True, fig=None, row: int = None):
            colorscale = [
                [0.0, '#ebedf0'],
                [1.0, '#F6339A']
            ]
            
            # Rest of the function remains the same as display_year
            if year is None:
                year = datetime.now().year
                
            d1 = datetime(year, 1, 1)
            d2 = datetime(year, 12, 31)

            number_of_days = (d2 - d1).days + 1
            
            data = z  # z is already padded to the full year length

            delta = d2 - d1

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_days =   [31,    28,    31,     30,    31,     30, 
                           31,    31,    30,    31,    30,    31]
            if number_of_days == 366:  # leap year
                month_days[1] = 29
            month_positions = (np.cumsum(month_days) - 15)/7

            dates_in_year = [d1 + timedelta(days=i) for i in range(delta.days + 1)]
            weekdays_in_year = [date.weekday() for date in dates_in_year]  # 0=Monday

            weeknumber_of_dates = [date.isocalendar().week for date in dates_in_year]
            # Handle edge cases where ISO week might be 53 or 0
            weeknumber_of_dates = [w if w <= 53 else 53 for w in weeknumber_of_dates]

            text = [str(date.date()) for date in dates_in_year]  # Hover text
            
            heatmap = go.Heatmap(
                x=weeknumber_of_dates,
                y=weekdays_in_year,
                z=data,
                text=text,
                hoverinfo='text',
                xgap=3,  # Gap between cells
                ygap=3,  # Gap between cells
                showscale=False,
                colorscale=colorscale,
                zmin=0,
                zmax=max(data) if max(data) > 0 else 1  # Dynamic zmax based on data
            )
            
            data_traces = [heatmap]
                
            if month_lines:
                kwargs = dict(
                    mode='lines',
                    line=dict(
                        color='#9e9e9e',
                        width=1,
                    ),
                    hoverinfo='skip',
                )
                
                for date, dow, wkn in zip(dates_in_year, weekdays_in_year, weeknumber_of_dates):
                    if date.day == 1:
                        data_traces += [
                            go.Scatter(
                                x=[wkn - 0.5, wkn - 0.5],
                                y=[dow - 0.5, 6.5],
                                **kwargs,
                            )
                        ]
                        if dow:
                            data_traces += [
                                go.Scatter(
                                    x=[wkn - 0.5, wkn + 0.5],
                                    y=[dow - 0.5, dow - 0.5],
                                    **kwargs,
                                ),
                                go.Scatter(
                                    x=[wkn + 0.5, wkn + 0.5],
                                    y=[dow - 0.5, -0.5],
                                    **kwargs,
                                )
                            ]
                            
            layout = go.Layout(
                height=250,
                yaxis=dict(
                    showline=False, showgrid=False, zeroline=False,
                    tickmode='array',
                    ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    tickvals=[0, 1, 2, 3, 4, 5, 6],
                    autorange="reversed",
                ),
                xaxis=dict(
                    showline=False, showgrid=False, zeroline=False,
                    tickmode='array',
                    ticktext=month_names,
                    tickvals=month_positions,
                ),
                font={'size':10, 'color':'#9e9e9e'},
                plot_bgcolor=('#fff'),
                margin = dict(t=40, l=20, r=20, b=20),
                showlegend=False,
            )

            if fig is None:
                fig = go.Figure(data=data_traces, layout=layout)
            else:
                fig.add_traces(data_traces, rows=[(row+1)]*len(data_traces), cols=[1]*len(data_traces))
                fig.update_layout(layout)
                fig.update_xaxes(layout['xaxis'])
                fig.update_yaxes(layout['yaxis'])

            return fig
        
        # Use the custom display_years function with the custom color
        def custom_display_years(z_yearly, years):
            fig = make_subplots(
                rows=len(years), 
                cols=1, 
                subplot_titles=[f"Year {year}" for year in years], 
                vertical_spacing=0.05,
                # Set a reasonable aspect ratio that maintains square cells
                specs=[[{"type": "xy"}] for _ in years]
            )
            
            for i, (z, year) in enumerate(zip(z_yearly, years)):
                fig = custom_display_year(z, year=year, fig=fig, row=i)
            
            # Update layout with dynamic height calculation
            # Base height of 150px per year, with some padding
            fig.update_layout(
                height=max(250 * len(years), 250),  # Minimum height of 250px
                # Force subplot y-axes to have 1:1 aspect ratio with x-axes
                yaxis=dict(scaleanchor="x", scaleratio=1),
                yaxis2=dict(scaleanchor="x2", scaleratio=1)
            )
            
            return fig
        
        fig = custom_display_years(z_yearly, years_to_display)
        st.plotly_chart(fig, use_container_width=True)

        # Show the raw data in an expandable section
        with st.expander("View Raw Data"):
            st.dataframe(sex_df.sort_values('Start', ascending=False))
    else:
        st.write("No intimacy events found.")

def main():
    # Set page configuration
    st.set_page_config(page_title="Calendar App", layout="wide")
    
    # Create page selection
    page = st.sidebar.radio("Select a Page", ["Home", "Exercise", "Alcohol", "Intimacy"])
    
    # Render selected page
    if page == "Home":
        home_page()
    elif page == "Exercise":
        exercise_page()
    elif page == "Alcohol":
        alcohol_page()
    elif page == "Intimacy":
        sex_page()

if __name__ == "__main__":
    main()