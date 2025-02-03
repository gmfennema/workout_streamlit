import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo
import re
import markdown

# Constants
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def clean_html_description(desc):
    """
    Remove HTML tags from the description and return plain text.
    """
    if not desc:
        return ""
    soup = BeautifulSoup(desc, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def get_calendar_service():
    """Authenticate and return the Google Calendar API service using environment variables."""
    creds = Credentials.from_authorized_user_info({
        "token": st.secrets["google_token"],
        "refresh_token": st.secrets["google_refresh_token"],
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": st.secrets["google_client_id"],
        "client_secret": st.secrets["google_client_secret"],
        "scopes": SCOPES
    })
    
    return build('calendar', 'v3', credentials=creds)

@st.cache_data
def get_events(days=180):
    """
    Fetch events from the specified number of days ago through current date.
    Uses Streamlit caching to avoid repeated API calls.
    
    Args:
        days (int): Number of days to fetch events for. Defaults to 365.
    """
    service = get_calendar_service()
    all_events = []
    
    # Get current time
    now = datetime.now(timezone.utc)
    end_time = now
    
    # Start from specified days ago
    current_start = now - timedelta(days=days)
    # Use 30-day chunks to fetch events
    chunk_size = timedelta(days=30)
    
    while current_start < end_time:
        current_end = min(current_start + chunk_size, end_time)
        
        # Format for RFC3339 timestamp
        start_str = current_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = current_end.strftime('%Y-%m-%dT%H:%M:%SZ')

        try:
            events_result = service.events().list(
                calendarId='primary',
                timeMin=start_str,
                timeMax=end_str,
                maxResults=2500,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            chunk_events = events_result.get('items', [])
            all_events.extend(chunk_events)
            
        except Exception as e:
            st.error(f"Error fetching events: {e}")
        
        # Move the window forward
        current_start = current_end

    return all_events

def clean_html_description(html_text):
    """Clean HTML from description and return structured text."""
    if not html_text:
        return ''
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Handle paragraphs and lists
        for elem in soup.find_all(['p', 'ul']):
            # Remove dir="auto" attributes
            del elem['dir']
            
            # Add newlines around paragraphs and lists
            elem.insert_before('\n')
            elem.insert_after('\n')
        
        # Handle list items
        for li in soup.find_all('li'):
            del li['dir']
            # Preserve the list item marker and add proper spacing
            li.insert_before('• ')
            li.insert_after('\n')
        
        # Get text and clean up whitespace
        text = soup.get_text()
        
        # Clean up excessive whitespace and newlines while preserving structure
        lines = []
        for line in text.split('\n'):
            cleaned_line = ' '.join(line.split())
            if cleaned_line:
                # Preserve bullet points formatting
                if cleaned_line.startswith('•'):
                    cleaned_line = cleaned_line.replace('• ', '• ', 1)  # Ensure consistent bullet spacing
                lines.append(cleaned_line)
        
        return '\n'.join(lines)
    except:
        return html_text.strip()

def process_events(events):
    """Process raw calendar events into a DataFrame."""
    processed_events = []
    
    for event in events:
        # Check if the event has a specific time (not an all-day event)
        start = event['start'].get('dateTime')
        end = event['end'].get('dateTime')
        
        # Skip all-day events
        if not start or not end:
            continue

        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        # Convert to Mountain Time (America/Phoenix)
        mt_tz = ZoneInfo("America/Phoenix")
        start_mt = start_dt.astimezone(mt_tz)
        end_mt = end_dt.astimezone(mt_tz)
        
        title = event.get('summary', 'No Title')
        color = event.get('colorId', '0')
        event_id = event.get('id', 'No ID')
        description = clean_html_description(event.get('description', ''))
        
        processed_events.append({
            'Event ID': event_id,
            'Start': start_mt,
            'End': end_mt,
            'Title': title,
            'Description': description,
            'Color ID': color
        })
    
    return pd.DataFrame(processed_events)

def get_category(color_id):
    """Map color ID to category."""
    category_map = {
        '0': 'Family Time',
        '1': 'Social',
        '2': 'Navigators',
        '3': 'Chores',
        '4': 'Ignore',
        '5': 'Spiritual',
        '6': 'N/A',
        '7': 'Consulting',
        '8': 'Personal',
        '9': 'Exercise'
    }
    return category_map.get(str(color_id), f'Unknown: {color_id}')

def display_year_flipped(z, types, exercised_with, descriptions, year: int = None, month_lines: bool = True, fig=None, row: int = None):
    """
    Display the exercise heatmap for a given year with weeks on Y-axis and weekdays on X-axis.
    """
    if year is None:
        year = datetime.now().year

    d1 = datetime(year, 1, 1)
    d2 = datetime(year, 12, 31)

    number_of_days = (d2 - d1).days + 1
    data = z  

    dates_in_year = [d1 + timedelta(days=i) for i in range(number_of_days)]
    weekdays_in_year = [date.weekday() for date in dates_in_year]  # 0=Monday
    weeknumber_of_dates = [date.isocalendar()[1] for date in dates_in_year]
    weeknumber_of_dates = [w if w <= 53 else 53 for w in weeknumber_of_dates]

    binary_data = (np.array(data) > 0).astype(int)
    date_strings = [date.strftime("%A, %b %d, %Y") for date in dates_in_year]

    # Updated helper to extract "Muscle Group:" information without the duplicate prefix.
    def extract_muscle_group(desc):
        for line in str(desc).splitlines():
            stripped = line.strip()
            if stripped.startswith("Muscle Group:"):
                # Remove the prefix so that the hover template doesn't duplicate.
                return stripped[len("Muscle Group:"):].strip()
        return ""
    
    muscle_group_lines = [extract_muscle_group(desc) for desc in descriptions]

    # Pack the date, type, exercised with, and muscle group info into custom data.
    custom_data = list(zip(date_strings, types, exercised_with, muscle_group_lines))

    heatmap = go.Heatmap(
        x=weekdays_in_year,
        y=weeknumber_of_dates,
        z=binary_data,
        customdata=custom_data,
        hovertemplate=("Date: %{customdata[0]}<br>"
                       "Type: %{customdata[1]}<br>"
                       "Exercised with: %{customdata[2]}<br>"
                       "Muscle Group: %{customdata[3]}<extra></extra>"),
        xgap=3,
        ygap=3,
        showscale=False,
        colorscale=[
            [0.0, '#ebedf0'],
            [1.0, '#02A6F4']
        ],
        zmin=0,
        zmax=1
    )
    data_traces = [heatmap]

    if month_lines:
        kwargs = dict(
            mode='lines',
            line=dict(color='#9e9e9e', width=1),
            hoverinfo='skip',
        )
        for date, dow, wkn in zip(dates_in_year, weekdays_in_year, weeknumber_of_dates):
            if date.day == 1:
                data_traces += [
                    go.Scatter(
                        y=[wkn - 0.5, wkn - 0.5],
                        x=[dow - 0.5, 6.5],
                        **kwargs,
                    )
                ]
                if dow:
                    data_traces += [
                        go.Scatter(
                            y=[wkn - 0.5, wkn + 0.5],
                            x=[dow - 0.5, dow - 0.5],
                            **kwargs,
                        ),
                        go.Scatter(
                            y=[wkn + 0.5, wkn + 0.5],
                            x=[dow - 0.5, -0.5],
                            **kwargs,
                        )
                    ]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if number_of_days == 366:  # leap year
        month_days[1] = 29
    month_positions = (np.cumsum(month_days) - 15) / 7

    layout = go.Layout(
        height=1800,
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            side='top',
            automargin=True,
            tickmode='array',
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticklabelstandoff=2,
            position=0.95,
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            tickmode='array',
            ticktext=month_names,
            tickvals=month_positions,
            autorange="reversed",
        ),
        font={'size': 10, 'color': '#9e9e9e'},
        plot_bgcolor='#fff',
        margin=dict(t=80, l=80, r=20, b=40),
        showlegend=False,
    )

    if fig is None:
        fig = go.Figure(data=data_traces, layout=layout)
    else:
        row_index = row + 1 if row is not None else 1
        fig.add_traces(data_traces, rows=[row_index] * len(data_traces), cols=[1] * len(data_traces))
        fig.update_layout(layout)

    fig.update_layout(
        hoverlabel=dict(
            align='left',
            font_size=12,
            font_family="sans-serif",
            bgcolor="white"
        )
    )
    return fig

def display_years_flipped(z_yearly, types_yearly, exercised_with_yearly, descriptions_yearly, years):
    """
    Display multiple years of exercise heatmaps with weekdays on X-axis and months on Y-axis.
    """
    fig = make_subplots(
        rows=len(years), 
        cols=1, 
        subplot_titles=[f"Year {year}" for year in years], 
        vertical_spacing=.55,
        specs=[[{"type": "xy"}] for _ in years]
    )
    
    for i, (z, typ, ex_with, desc, year) in enumerate(zip(z_yearly, types_yearly, exercised_with_yearly, descriptions_yearly, years)):
        fig = display_year_flipped(z, typ, ex_with, desc, year=year, fig=fig, row=i)
    
    fig.update_layout(
        height=2000 * len(years),  # Fixed height per year
        showlegend=False,
        margin=dict(t=100, l=60, r=80, b=50)
    )
    
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.update(y=ann.y - 0.05)
    
    return fig

def format_workout_description(description: str) -> str:
    """
    Format workout description into clean Markdown sections.
    """
    if not description:
        return ""
    
    # Split into lines and clean up
    lines = [line.strip() for line in description.split('\n') if line.strip()]
    formatted_lines = []
    current_section = None
    
    for line in lines:
        # Handle main sections
        if any(line.startswith(prefix) for prefix in ['Muscle Group:', 'RPE:', 'Notes:']):
            formatted_lines.extend(['', f"### {line}", ''])
            current_section = 'main'
        
        # Handle exercise names
        elif line.endswith(':'):
            formatted_lines.extend(['', f"#### {line.rstrip(':')}", ''])
            current_section = 'exercise'
        
        # Handle bullet points
        elif line.startswith('•'):
            clean_line = line[1:].strip()  # Remove bullet and clean
            # Split long exercise lines into multiple lines for readability
            sets = clean_line.split('•')
            for set_info in sets:
                if set_info.strip():
                    formatted_lines.append(f"- {set_info.strip()}")
        
        # Handle notes without bullets
        elif current_section == 'main':
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def main():
    st.set_page_config(page_title="Workout Tracker", layout="centered")  # Centered layout for fixed width
    
    st.markdown("""
        <style>
        /* Container adjustments */
        .stApp {
            max-width: 450px !important;
            margin: 20px auto !important;
            padding: 0 10px;
        }
        /* Center all elements within the chart */
        .stPlotlyChart > div {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        /* Hide fullscreen button on mobile */
        .modebar-btn[data-title="Toggle Fullscreen"] {
            display: none !important;
        }
        /* Reduce title size */
        h1, h2, h3, h4, h5, h6 {
            font-size: 1.5rem !important;
        }
        /* Adjust metrics grid for mobile */
        @media (max-width: 450px) {
            .metrics-grid {
                flex-direction: column;
            }
            .metrics-grid > div {
                flex-basis: 100% !important;
                margin: 5px 0 !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Exercise Tracker")

    # Fetch and process events
    events = get_events()
    df = process_events(events)
    
    # Filter for Exercise events
    df['Category'] = df['Color ID'].apply(get_category)
    exercise_df = df[df['Category'] == 'Exercise'].copy()
    
    if not exercise_df.empty:
        # Define the target lifts
        target_lifts = ["Deadlift", "Bench Press", "Squat"]
        # Dictionary to hold the most recent PR weight and date for each lift.
        pr_dict = {lift: None for lift in target_lifts}
        
        # Sort by the start time descending so the first occurrence is the most recent.
        for _, row in exercise_df.sort_values('Start', ascending=False).iterrows():
            description = row['Description']
            if "PR |" in description:
                for line in description.splitlines():
                    if line.startswith("PR |"):
                        # Match pattern like "PR | Squat | 330 lbs"
                        m = re.match(r"PR\s*\|\s*([^|]+)\s*\|\s*([\d\.]+\s*lbs)", line)
                        if m:
                            lift = m.group(1).strip()
                            weight = m.group(2).strip()
                            if lift in target_lifts and pr_dict[lift] is None:
                                # Save both the weight and the event's Start date
                                pr_dict[lift] = {"weight": weight, "date": row['Start']}
            # If we've found a PR for all target lifts, we can break early.
            if all(pr_dict[lift] is not None for lift in target_lifts):
                break

        st.subheader("Personal Records")
        pr_html = "<div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>"
        for lift in target_lifts:
            record = pr_dict[lift]
            if record is not None:
                result = record["weight"]
                # Format the date as "YYYY-MM-DD"
                date_str = record["date"].strftime("%Y-%m-%d") if record.get("date") else ""
            else:
                result = "N/A"
                date_str = ""
            pr_html += (
                f"<div style='background-color: white; border-radius: 8px; padding: 10px; margin: 5px; "
                "flex-basis: calc(33.33% - 10px); text-align: center; "
                "box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
                f"<p style='margin-bottom: 5px; color: #333; font-weight: bold;'>{lift}</p>"
                f"<div style='font-size: 24px; font-weight: bold; color: #02A6F4;'>{result}</div>"
            )
            if date_str:
                pr_html += f"<div style='font-size: 12px; font-weight: 300; color:rgb(171, 172, 173);'>{date_str}</div>"
            pr_html += "</div>"
        pr_html += "</div>"
        st.markdown(pr_html, unsafe_allow_html=True)
        
        st.divider()
        
        # Muscle Group Filter for the activity section with capitalized options
        selected_muscle_group = st.selectbox(
            "Filter by Muscle Group", 
            ["All", "Chest", "Back", "Legs", "Biceps", "Triceps", "Core"]
        )
        if selected_muscle_group != "All":
            filtered_exercise_df = exercise_df[
                exercise_df['Description'].str.lower().str.contains(selected_muscle_group.lower())
            ]
        else:
            filtered_exercise_df = exercise_df.copy()

        # New Text Search Filter for Exercise.
        exercise_search_term = st.text_input("Filter by Exercise", "")
        if exercise_search_term:
            filtered_exercise_df = filtered_exercise_df[
                filtered_exercise_df['Description'].str.lower().str.contains(exercise_search_term.lower())
            ]

        st.divider()
        st.subheader("Exercise Activity")
        
        if filtered_exercise_df.empty:
            st.write(f"No exercise events found for muscle group '{selected_muscle_group}'.")
        else:
            # Get all workout dates from filtered data
            workout_dates = filtered_exercise_df['Start'].dt.date.unique()
            workout_dates = sorted(workout_dates, reverse=True)
            
            # Add a default option for no date selected
            options = [None] + list(workout_dates)
            selected_date = st.selectbox(
                "Select a date to view workout details",
                options=options,
                format_func=lambda x: x.strftime('%Y-%m-%d') if x is not None else "Select a date"
            )
            
            # Display workout details if a valid date is selected
            if selected_date is not None:
                day_workouts = filtered_exercise_df[
                    filtered_exercise_df['Start'].dt.date == selected_date
                ]
                descriptions = day_workouts['Description'].dropna().tolist()
                
                if descriptions:
                    # First, convert each detail using your formatting function.
                    formatted_descriptions = [format_workout_description(desc) for desc in descriptions]
                    
                    # Combine all formatted markdown into one string
                    combined_md = "\n\n".join(formatted_descriptions)
                    # Convert the markdown string to HTML
                    html_converted = markdown.markdown(combined_md)
                    # Wrap the HTML in a styled container
                    styled_html = f"""
                    <div style="background-color: #f8f8f8; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px 20px 10px 40px;">
                        {html_converted}
                    </div>
                    """
                    
                    st.markdown(f"### {selected_date} Workout Details")
                    st.markdown("---")
                    st.markdown(styled_html, unsafe_allow_html=True)
            
            # Prepare and display the heatmap
            current_year = datetime.now(ZoneInfo("America/Phoenix")).year
            years_to_display = [current_year]
            z_yearly = []
            types_yearly = []
            exercised_with_yearly = []
            descriptions_yearly = []
            
            for target_year in years_to_display:
                start_date = datetime(target_year, 1, 1).date()
                end_date = datetime.now(ZoneInfo("America/Phoenix")).date()
                
                date_range = pd.date_range(start=start_date, end=end_date)
                
                # Aggregate daily events: count, concatenated descriptions, and first event title.
                daily_data = filtered_exercise_df[
                    (filtered_exercise_df['Start'].dt.date >= start_date) & 
                    (filtered_exercise_df['Start'].dt.date <= end_date)
                ].groupby(filtered_exercise_df['Start'].dt.date).agg({
                    'Start': 'count',
                    'Description': lambda x: '\n'.join(filter(None, x)) if any(filter(None, x)) else '',
                    'Title': 'first'
                })
                daily_data = daily_data.reindex(date_range.date)
                daily_data['Start'].fillna(0, inplace=True)
                daily_data['Description'].fillna('', inplace=True)
                daily_data['Title'].fillna('', inplace=True)
                
                z = daily_data['Start'].values
                descriptions_arr = daily_data['Description'].values
                titles_arr = daily_data['Title'].values
                
                def get_exercise_type(title):
                    return title.split('|')[0].strip() if '|' in title else title.strip()
                
                def get_exercised_with(title):
                    if '|' in title:
                        parts = title.split('|', 1)
                        value = parts[1].strip()
                        return value if value else "Alone"
                    else:
                        return "Alone"
                
                types_arr = np.array([get_exercise_type(t) for t in titles_arr])
                exercised_with_arr = np.array([get_exercised_with(t) for t in titles_arr])
                
                full_year_days = (datetime(target_year, 12, 31) - datetime(target_year, 1, 1)).days + 1
                if len(z) < full_year_days:
                    z = np.concatenate([z, np.zeros(full_year_days - len(z))])
                    descriptions_arr = np.concatenate([descriptions_arr, np.array([''] * (full_year_days - len(descriptions_arr)))])
                    types_arr = np.concatenate([types_arr, np.array([''] * (full_year_days - len(types_arr)))])
                    exercised_with_arr = np.concatenate([exercised_with_arr, np.array(['Alone'] * (full_year_days - len(exercised_with_arr)))])
                elif len(z) > full_year_days:
                    z = z[:full_year_days]
                    descriptions_arr = descriptions_arr[:full_year_days]
                    types_arr = types_arr[:full_year_days]
                    exercised_with_arr = exercised_with_arr[:full_year_days]
                
                z_yearly.append(z)
                descriptions_yearly.append(descriptions_arr)
                types_yearly.append(types_arr)
                exercised_with_yearly.append(exercised_with_arr)
            
            fig = display_years_flipped(z_yearly, types_yearly, exercised_with_yearly, descriptions_yearly, years_to_display)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with st.expander("View Raw Data"):
                st.dataframe(filtered_exercise_df.sort_values('Start', ascending=False))
    else:
        st.write("No exercise events found.")

if __name__ == "__main__":
    main()