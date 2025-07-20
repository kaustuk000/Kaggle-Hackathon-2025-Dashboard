import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import folium


# Load data
comp_per_year = pd.read_csv("comp_per_year.csv", index_col=0)
prized_comp_yearly = pd.read_csv("prized_comp_yearly.csv")
df_ai_invest = pd.read_csv("df_ai_invest.csv")
sector_year_counts = pd.read_csv("sector_year_counts.csv", index_col=0)
df_ai_invest_indwise_count = pd.read_csv("df_ai_invest_indwise_count.csv", index_col=0)
df_ai_research_count = pd.read_csv("df_ai_research_count.csv")
corr_invest_selected = pd.read_csv("corr_invest_selected.csv")
corr_pubs_selected = pd.read_csv("corr_pubs_selected.csv")
df_prized_problem_type = pd.read_csv("df_prized_problem_type.csv")
alice_mask = np.array(Image.open('alice_mask.png'))
word_freq_df = pd.read_csv("word_freq.csv")
word_freq_df.columns = ['word', 'frequency']
top20_gm_earlier = pd.read_csv("top20_gm_earlier.csv")
top20_gm_recent = pd.read_csv("top20_gm_recent.csv")
pivot_summary = pd.read_csv("pivot_summary.csv")
users_map = pd.read_csv("users_map.csv")
advance_map = pd.read_csv("advance_map.csv")
growth_map = pd.read_csv("growth_map.csv")
heatmap_data = pd.read_csv("heatmap_data.csv")
yearly_dataset_count = pd.read_csv("yearly_dataset_count.csv")
engagement_dataset_avg = pd.read_csv("engagement_dataset_avg.csv")
yearly_kernel_count = pd.read_csv("yearly_kernel_count.csv")
engagement_kernel_avg = pd.read_csv("engagement_kernel_avg.csv")
percent = pd.read_csv("percent.csv")
yearly_practice_counts = pd.read_csv("yearly_practice_counts.csv")
user_practice_per_year = pd.read_csv("user_practice_per_year.csv")

word_freq_dict = dict(zip(word_freq_df['word'], word_freq_df['frequency']))






indrustrial_sectors = sector_year_counts.drop(columns = ['Education','Enviroment and Earth','Research',
                                                         'Sports & Games'])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Kaggle Hackathon 2025 Dashboard"

def create_podium_placement_chart(pivot_summary):
    fig = go.Figure()

    colors = {
        "Advanced(GrandMasters & Masters)": "#1f77b4",
        "Intermediate(Experts)": "#ff7f0e",
        "Beginner(Novice & Contributer)": "#2ca02c"
    }

    cumulative = [0] * len(pivot_summary)

    for tier in ["Advanced(GrandMasters & Masters)", "Intermediate(Experts)", "Beginner(Novice & Contributer)"]:
        fig.add_trace(go.Bar(
            x=pivot_summary["PublicLeaderboardRank"],
            y=pivot_summary[tier],
            name=tier,
            marker_color=colors[tier],
            text=pivot_summary[tier],
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        title="Podium Placements by Tier Group (2014–2024)",
        xaxis_title="Leaderboard Rank (1 = First Place, etc.)",
        yaxis_title="Number of Placements",
        legend_title="Tier Group",
        template="plotly_white",
        margin=dict(t=60, b=40),
        height=500
    )

    return fig


# For Creating folium maps
def render_map_from_df(df, color = "blue", fill_color = "blue", GrowthMap = False):
    # Create folium map
    m = folium.Map(location=[20, 0], zoom_start=2)

    if not GrowthMap:
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=row["Radius"],
                color= color,
                fill=True,
                fill_color= fill_color,
                fill_opacity=0.6,
                popup=f"{row['Country']}: {int(row['UserCount'])} users"
            ).add_to(m)
    else:
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=row["ScaledRadius"],
                color="green" if row["Change"] > 0 else "red",
                fill=True,
                fill_color="green" if row["Change"] > 0 else "red",
                fill_opacity=0.6,
                popup=f"{row['Country']}: {'+' if row['Change'] > 0 else ''}{row['Change']} users"   
            ).add_to(m)
    # Return HTML string
    return m.get_root().render()





app.layout = html.Div([
    html.H1("Dashboard for Kaggle Hackathon 2025", className="text-center my-4"),
    html.Hr(),
    html.Div([
        html.H2("When Kaggle Met AI: A Journey of Mutual Growth", className="text-primary mt-4"),

    html.P(
        "For 15 years, Kaggle’s community has driven advances in AI and ML through real‑world competitions. "
        "To celebrate the launch of Kaggle Hackathons, this challenge uses the Meta Kaggle and Meta Kaggle Code "
        "datasets to uncover how participants and AI have co‑evolved.", 
        className="mb-3"
    ),

    html.P(
        "By mining competition metadata and notebook code, we’ll reveal trends, patterns, and dynamics that "
        "showcase the mutual growth of Kaggle and the broader AI industry.",
        className="mb-4"
    ),

    html.H4("Task Given", className="text-success"),
    html.P(
        "Analyze the Meta Kaggle and Meta Kaggle Code datasets to uncover how Kaggle competitions and the community "
        "have shaped AI/ML progress over the past 15 years.",
        className="mb-4"
    ),

    html.H4("Our Analytical Journey: 3 Stages, 1 Story", className="text-primary"),
    html.P("To uncover how Kaggle and AI have shaped each other and grown along the way, our analysis unfolds in three thematic stages:"),
    html.Ul([
        html.Li("AI/ML’s Imprint on Kaggle"),
        html.Li("The Community’s Climb"),
        html.Li("Kaggle’s Mark on the AI World")
    ], className="mb-4"),

    html.H4("Exploratory Data Analysis for Kaggle Hackathon 2025", className="text-primary mt-4"),
        html.P("This dashboard presents insights derived from various stages of the Kaggle Hackathon 2025. "
               "Use the buttons below to navigate through the detailed analysis for each stage.")
    ], className="container mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Button("AI/ML’s Imprint on Kaggle", id="btn-stage1", color="primary", className="w-100 mb-2")),
            dbc.Col(dbc.Button("The Community’s Climb", id="btn-stage2", color="success", className="w-100 mb-2")),
            dbc.Col(dbc.Button("Kaggle’s Mark on the AI World", id="btn-stage3", color="warning", className="w-100 mb-2")),
        ], justify="center"),

        html.Hr(),
        html.Div(id="stage1-output1"),
        html.Br()
    ])
])

@callback(
    Output("stage1-output1", "children"),
    Input("btn-stage1", "n_clicks"),
    Input("btn-stage2", "n_clicks"),
    Input("btn-stage3", "n_clicks")
)
def render_stage(btn1, btn2, btn3):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if clicked_id == "btn-stage1":
        # STAGE 1 FIGURE 1 SUBPLOT
        stage1Fig1 = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Kaggle Competitions per Year",
                "Competitions by Top Host Segment",
                "Global AI Investment Over Years"
            )
        )

        # STAGE 1 SUBPLOT 2
        stage1Fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Kaggle Competitions by Sector Over the Years",
                "AI Investment Over the Year"
            )
        )
        

        # STAGE 1 SUBPLOT 3
        stage1Fig3 = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Kaggle Competitions On Research Sector",
                "Scientific publications on AI"
            )
        )
        
        # STAGE 1 SUBPLOT 4
        stage1Fig4 = make_subplots(rows=1, cols=2, subplot_titles=(
            "Lag Correlation: Competitions vs AI Investment",
            "Lag Correlation: Competitions vs AI Publications"
        ))


        # STAGE 1 SUBPLOT 1 PLOTTING
        stage1Fig1.add_trace(go.Scatter(
            x=comp_per_year.index, y=comp_per_year.values.flatten(), name="Competitions", mode='lines+markers'
        ), row=1, col=1)
        stage1Fig1.update_xaxes(title_text="Year", row=1, col=1)
        stage1Fig1.update_yaxes(title_text="Number of Competitions", row=1, col=1)

        stage1Fig1.add_trace(go.Scatter(
            x=prized_comp_yearly['YearHosted'], y=prized_comp_yearly['PrizeCompetitionsCount'],
            mode='lines+markers', name="Prized Comps", line=dict(color='darkred')
        ), row=1, col=2)
        stage1Fig1.update_xaxes(title_text="Year", row=1, col=2)
        stage1Fig1.update_yaxes(title_text="Number of Prized Competitions", row=1, col=2)

        stage1Fig1.add_trace(go.Scatter(
            x=df_ai_invest['Year'], y=df_ai_invest['Sum_of_deals'],
            mode='lines+markers', name="AI Investment", line=dict(color='green')
        ), row=1, col=3)
        stage1Fig1.update_xaxes(title_text="Year", row=1, col=3)
        stage1Fig1.update_yaxes(title_text="Sum of Deals", row=1, col=3)


        # STAGE 1 SUBPLOT 2 PLOTTING
        for col in indrustrial_sectors.columns:
            stage1Fig2.add_trace(go.Scatter(
                x=sector_year_counts.index, y=sector_year_counts[col], mode='lines+markers', name=col
            ), row=1, col=1)
        stage1Fig2.update_xaxes(title_text="Year", row=1, col=1)
        stage1Fig2.update_yaxes(title_text="No. of Competitions", row=1, col=1)

        for col in df_ai_invest_indwise_count.columns:
            stage1Fig2.add_trace(go.Scatter(
                x=df_ai_invest_indwise_count.index, y=df_ai_invest_indwise_count[col], mode='lines+markers', name=col
            ), row=2, col=1)
        stage1Fig2.update_xaxes(title_text="Year", row=2, col=1)
        stage1Fig2.update_yaxes(title_text="Investment Amount", row=2, col=1)


        # STAGE 1 SUBPLOT 3 PLOTTING
        stage1Fig3.add_trace(go.Scatter(x = sector_year_counts.index, y = sector_year_counts['Research'], mode = 'lines+markers', name = 'Research' ),
                             row = 1, col = 1)
        stage1Fig3.add_trace(go.Scatter(x = df_ai_research_count['year'], y = df_ai_research_count['Artificial Intelligence'], mode = 'lines+markers', name = "Artificial Intelligence"),
                             row = 2, col = 1)
        stage1Fig3.update_xaxes(title_text = "Year", row = 1 , col = 1)
        stage1Fig3.update_yaxes(title_text = "No. of Competitions", row = 1 , col = 1)
        
        stage1Fig3.update_xaxes(title_text = "Year", row = 2 , col = 1)
        stage1Fig3.update_yaxes(title_text = "Number of Publication", row = 2 , col = 1)


        # STAGE 1 SUBPLOT 4 PLOTTING
        
        # Define sector names in the correct order
        sectors = [
            'Healthcare, drugs and biotechnology',
            'IT infrastructure and hosting',
            'Business processes and support services',
            'Mobility and autonomous vehicles',
            'Media, social platforms, marketing'
        ]

        # Plot: Investment Correlation 
        stage1Fig4.add_trace(go.Bar(
            x=sectors,
            y=corr_invest_selected['1-Year Lag'],
            name="1-Year Lag(Investment)",
            marker_color='#1f77b4'
        ), row=1, col=1)

        stage1Fig4.add_trace(go.Bar(
            x=sectors,
            y=corr_invest_selected['2-Year Lag'],
            name="2-Year Lag(Investment)",
            marker_color='#ff7f0e'
        ), row=1, col=1)

        # Plot: Publications Correlation 
        stage1Fig4.add_trace(go.Bar(
            x=sectors,
            y=corr_pubs_selected['1-Year Lag'],
            name="1-Year Lag(Publication)",
            marker_color='green',
        ), row=1, col=2)

        stage1Fig4.add_trace(go.Bar(
            x=sectors,
            y=corr_pubs_selected['2-Year Lag'],
            name="2-Year Lag(Publication)",
            marker_color='red',
        ), row=1, col=2)

        # Axis formatting
        stage1Fig4.update_xaxes( tickangle=45, row=1, col=1)
        stage1Fig4.update_xaxes( tickangle=45, row=1, col=2)
        stage1Fig4.update_yaxes(title_text="Correlation", row=1, col=1)
        stage1Fig4.update_yaxes(title_text="Correlation", row=1, col=2)
        stage1Fig4.update_layout(barmode='group', height=500)
            
       
       # STAGE 1 SUBPLOT 5: Classical ML vs LLM Focus Shift


        years = df_prized_problem_type['YearHosted']
        classical_counts = df_prized_problem_type['Classical ML Based']
        llm_counts = df_prized_problem_type['LLM Based']

        stage1Fig5 = go.Figure()

        stage1Fig5.add_trace(go.Bar(
            x=years, y=classical_counts, name="Classical ML Based", marker_color='#1f77b4'
        ))

        stage1Fig5.add_trace(go.Bar(
            x=years, y= llm_counts, name="LLM Based", marker_color='#ff7f0e'
        ))

        # Vertical line for 2020
        stage1Fig5.add_vline(
            x=2020 - 0.5, line_dash="dash", line_color="red", line_width=2,
            annotation_text="LLM Influence Begins (2020)", annotation_position="top left"
        )

        # Vertical line for 2023
        stage1Fig5.add_vline(
            x=2023 - 0.5, line_dash="dot", line_color="purple", line_width=2,
            annotation_text="Mainstream LLM Adoption (2023)", annotation_position="top right"
        )

        stage1Fig5.update_layout(
            title="Focus Shift: Classical ML vs LLM/GenAI on Kaggle (2014–2024)",
            xaxis_title="Year",
            yaxis_title="Number of Prized Competitions",
            barmode='group',
            legend_title="Competition Type",
            height=500
        )

        # Generate word cloud
        wordcloud = WordCloud(
            background_color='white',
            width=800,
            height=600,
            mask= alice_mask,
            colormap='viridis'
        ).generate_from_frequencies(word_freq_dict)

        # Convert to PNG image for embedding
        img_buf = BytesIO()
        wordcloud.to_image().save(img_buf, format='PNG')
        img_buf.seek(0)
        encoded_wc = base64.b64encode(img_buf.read()).decode('utf-8')


        return html.Div([
            html.H4("Stage 1: How AI/ML Shaped Kaggle", className="text-info mt-4"),
            
            html.P(
                "From 2014 to 2024, Kaggle transformed from a data‑science collaboration site into a barometer of AI/ML trends. "
                "In this stage, we chart how competition numbers, problem domains, and prize pools have moved in step with global AI "
                "investment, research output, and the rise of LLMs and generative models."
            ),
            html.H5("Insights Covered in Stage 1", className="text-secondary mt-4"),
            html.Ul([
                html.Li("📈 Growth of Kaggle Competitions vs AI Investment (2014–2024) — How has competition frequency, especially high‑prize contests, evolved alongside global AI funding?"),
                html.Li("🏭 Sector‑Wise Growth of Competitions vs Investment — Which AI domains (healthcare, IT infra, mobility, etc.) have seen parallel trends in Kaggle and global AI investments?"),
                html.Li("🔮 Do Publications & Investments Predict Future Contests? — Does a surge in AI research papers or funding today foreshadow more Kaggle competitions 1–2 years down the line?"),
                html.Li("🤖 Impact of LLMs & Generative AI — How have breakthroughs in large language models and generative techniques reshaped the problem statements on Kaggle?"),
                html.Li("☁️ Word Cloud of Competition Themes (2020–2024) — This word cloud captures recurring themes from Kaggle competition titles during 2020–2024."),

                html.H5("Growth of Kaggle Competitions vs AI Investment (2014–2024)", className="text-primary mt-4"),
                html.P("In this stage, we examine the evolution of Kaggle competitions, their hosts, and the relationship to broader AI investment trends. The charts below give insights into how the platform responded to global AI momentum."),
                dcc.Graph(figure=stage1Fig1, style={"height": "450px"}),
                html.Br(),
                html.H5("Sector‑Wise Growth of Competitions vs Investment", className="text-primary mt-4"),
                html.P("We now dive deeper into how specific sectors evolved over time and how investment varied across them."),
                dcc.Graph(figure=stage1Fig2, style={"height": "700px"}),
                html.Br(),
                html.H5("Do Publications & Investments Predict Future Contests?", className="text-primary mt-4"),
                html.P("We now examine how Research Sector has evolved both in kaggle competiton and in Publication"),
                dcc.Graph(figure= stage1Fig3, style={"height": "700px"}),
                html.Br(),
                html.P("This plot shows the lag correlation between Kaggle competitions and AI activity (investment and publications) across specific sectors. "
                    "It helps identify where competitions might be leading or trailing industry and academic trends."),
                dcc.Graph(figure=stage1Fig4, style={"height": "600px"}),
                html.Br(),
                html.H5("Impact of LLMs & Generative AI", className="text-primary mt-4"),
                html.P("This chart illustrates the shift in focus from classical ML to LLM/GenAI-based competitions on Kaggle over the past decade. "
                    "The vertical lines denote two key moments: when LLMs began influencing competitions (2020), and when they entered mainstream use (2023)."),
                dcc.Graph(figure=stage1Fig5, style={"height": "600px"}),
                html.Br(),
                html.H5("Word Cloud of Competition Themes (2020–2024)", className="text-primary mt-4"),
                html.P("6. This word cloud captures recurring themes from Kaggle competition titles during 2020–2024. "
                    "Words like 'prediction', 'classification', and 'LLM' reflect the evolving nature of tasks and technology."),
                html.Img(src='data:image/png;base64,{}'.format(encoded_wc),
                style={"width": "53%","height": "53%", "display": "block", "margin": "auto"}),
                html.H4("Concluding Stage 1: How AI/ML Shaped Kaggle ", className="text-success mt-5"),
                html.P("Stage 1 concludes with a comprehensive overview of how Kaggle competitions have evolved in parallel with major trends in AI. "
                        "From shifts in investment and sector focus to the growing impact of LLMs and changing competition themes, "
                        "this stage sets the foundation for understanding the deeper dynamics explored in the upcoming stages."),    
            ])
        ],className="container mb-4")
    
    elif clicked_id == "btn-stage2":
        stage2Fig1 = make_subplots(
            rows=2, cols=2,
            
            horizontal_spacing=0.12,
            subplot_titles=(
                "Top 20 Recent Grandmasters",
                "Top 20 Early Grandmasters",
                "Regression: Recent Grandmasters",
                "Regression: Early Grandmasters"
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )

        # === Subplot 1: Recent GMs - Bar + Line
        x1 = list(top20_gm_recent['UserName'])
        stage2Fig1.add_trace(go.Bar(
            x=x1, y=top20_gm_recent['CompetitionsBeforeGM'],
            name="Competitions Before GM",
            marker_color="#1f77b4",
        ), row=1, col=1, secondary_y= False)

        stage2Fig1.add_trace(go.Scatter(
            x=x1, y=top20_gm_recent['YearsToGM'],
            name="Years to GM",
            mode='lines+markers',
            marker=dict(color="#ff7f0e"),
            yaxis="y2"
        ), row=1, col=1, secondary_y=True)

        # === Subplot 2: Early GMs - Bar + Line
        x2 = list(top20_gm_earlier['UserName'])
        stage2Fig1.add_trace(go.Bar(
            x=x2, y=top20_gm_earlier['CompetitionsBeforeGM'],
            name="Competitions Before GM",
            marker_color="#1f77b4",
            showlegend=False
        ), row=1, col=2,secondary_y= False)

        stage2Fig1.add_trace(go.Scatter(
            x=x2, y=top20_gm_earlier['YearsToGM'],
            name="Years to GM",
            mode='lines+markers',
            marker=dict(color="#ff7f0e"),
            yaxis="y4"
        ), row=1, col=2, secondary_y=True)

        # === Subplot 3: Regression - Recent GMs
        x3 = top20_gm_recent['CompetitionsBeforeGM']
        y3 = top20_gm_recent['YearsToGM']
        coef = np.polyfit(x3, y3, 1)
        trend = coef[0] * x3 + coef[1]

        stage2Fig1.add_trace(go.Scatter(
            x=x3, y=y3, mode='markers', name='Recent GM Data',
            marker=dict(color="#1f77b4", size=10, opacity=0.7)
        ), row=2, col=1)

        stage2Fig1.add_trace(go.Scatter(
            x=x3, y=trend, mode='lines', name='Trend Line',
            line=dict(color='black', dash='dash')
        ), row=2, col=1)

        # === Subplot 4: Regression - Early GMs
        x4 = top20_gm_earlier['CompetitionsBeforeGM']
        y4 = top20_gm_earlier['YearsToGM']
        coef2 = np.polyfit(x4, y4, 1)
        trend2 = coef2[0] * x4 + coef2[1]

        stage2Fig1.add_trace(go.Scatter(
            x=x4, y=y4, mode='markers', name='Early GM Data',
            marker=dict(color="#ff7f0e", size=10, opacity=0.7)
        ), row=2, col=2)

        stage2Fig1.add_trace(go.Scatter(
            x=x4, y=trend2, mode='lines', name='Trend Line',
            line=dict(color='black', dash='dash')
        ), row=2, col=2)

        # Update Layout
        stage2Fig1.update_layout(
            height=800,
            title="The Climb of Champions: Competitions vs Time to Reach Grandmaster",
            barmode="group",
            showlegend= True
        )

        # Axis formatting
        stage2Fig1.update_xaxes(title_text="User", tickangle=45, row=1, col=1)
        stage2Fig1.update_xaxes(title_text="User", tickangle=45, row=1, col=2)
        stage2Fig1.update_yaxes(title_text="Competitions Before GM", row=1, col=1, secondary_y=False)
        stage2Fig1.update_yaxes(title_text="Years to GM", row=1, col=1, secondary_y=True)
        stage2Fig1.update_yaxes(title_text="Competitions Before GM", row=1, col=2, secondary_y=False)
        stage2Fig1.update_yaxes(title_text="Years to GM", row=1, col=2, secondary_y=True)


        # STAGE 2 FIGURE 2 PLOTTING
        stage2Fig2 = create_podium_placement_chart(pivot_summary)

        # STAGE 2 MAP 1 PLOTTING DONE
        # STAGE 3 MAP 2 PLOTTING DONE

        # STAGE 2 FIGURE 3 PLOTTING
        countries = heatmap_data["Country"]
        years = heatmap_data.columns[1:]  # Exclude 'Country'
        z_values = heatmap_data[years].values
        
       # Create Heatmap
        stage2Fig3 = go.Figure(data=go.Heatmap(
            z=z_values,
            x=years,
            y=countries,
            colorscale="YlGnBu",
            text = z_values,
            texttemplate = "%{z}",
            textfont={"size": 10, "color": "black"}, 
            hovertemplate="Year: %{x}<br>Country: %{y}<br>Users: %{z}<extra></extra>",
            colorbar=dict(title="Users")
        ))

        stage2Fig3.update_layout(
        title="User Registrations by Country and Year (Top 10 Countries)",
        xaxis_title="Year",
        yaxis_title="Country",
        height=700,
        margin=dict(l=100, r=20, t=60, b=60)
    )

        # STAGE 2 FIGURE 4 PLOTTING
        # Melt for long format
        line_data = heatmap_data.melt(id_vars="Country", var_name="Year", value_name="Users")
        line_data["Year"] = line_data["Year"].astype(int)
        top_countries = heatmap_data.sort_values("2024", ascending=False).head(10)["Country"]
        line_data = line_data[line_data["Country"].isin(top_countries)]       
         
        stage2Fig4 = go.Figure()

        for country in top_countries:
            country_data = line_data[line_data["Country"] == country]
            stage2Fig4.add_trace(go.Scatter(
                x=country_data["Year"],
                y=country_data["Users"],
                mode="lines+markers",
                name=country
            ))

        stage2Fig4.update_layout(
        title="User Growth Trends Top 10 Countries(2014–2024)",
        xaxis_title="Year",
        yaxis_title="Number of Users",
        template="plotly_white",
        legend_title="Country",
        height=500,
        margin=dict(t=40, b=40)
    )



        return html.Div([
            html.H4("Stage 2: The Community's Climb", className="text-success mt-4"),
            
            html.P("As artificial intelligence matured, so did the Kaggle community — not just in size, "
                "but in skill, diversity, and contribution. This stage focuses on the people behind the kernels, "
                "the coders behind the models. It explores how Kagglers evolve, rise through the ranks, and shape "
                "the competitive landscape year after year."),

            html.H5("Insights Covered in Stage 2", className="text-secondary"),
            html.Ul([
                html.Li("🔺 The Climb of Champions — How do Kaggle legends rise to the top?"),
                html.Li("🧑‍💻 Do Beginners Win? — Can newcomers compete with experts?"),
                html.Li("🌍 Geographical Growth — Where is the Kaggle community growing fastest?"),
                html.Br(),
                html.H5("The Climb of Champions", className="text-primary mt-4"),
                html.P("How many competitions does it take to become a Grandmaster — and how long does it take? This visualization compares early and recent GrandMaster journeys, revealing evolving dynamics in the Kaggle elite tier."),
                dcc.Graph(figure=stage2Fig1, style={"width": "1300px"}),
                html.H5("Do Beginners Win?", className="text-primary mt-4"),
                html.P("Kagglers from different tiers have earned podiums over the years. This stacked bar chart shows how each tier has contributed to the top 3 leaderboard positions."),
                dcc.Graph(figure= stage2Fig2, style={"width": "100%", "overflowX": "auto"}),
                html.Br(),
                html.H5("Geographical Growth of the Kaggle Community" , className="text-primary mt-4"),
                html.B("1. Which countries are driving Kaggle’s global expansion?"),
                html.P("This visualization highlights the nations contributing the most in terms of user registrations, offering insight into where the most data science community is coming from."),

                html.Iframe(
                    srcDoc=render_map_from_df(users_map),
                    style={"width": "100%", "height": "600px", "border": "none"}
                ),
                html.Br(),
                html.Br(),
                html.B("2. Where is top-tier Kaggle talent concentrated?"),
                html.P("This visualization shows the global distribution of Grandmasters and Masters, revealing which countries are producing the highest density of elite Kagglers in the data science community."),
                html.Iframe(
                    srcDoc= render_map_from_df(df= advance_map, color= "darkred", fill_color="crimson"),
                    style={"width": "100%", "height" : "600px", "border": "none"}
                ),
                html.Br(),
                dcc.Graph(figure= stage2Fig3),
                html.Br(),
                dcc.Graph(figure= stage2Fig4),
                html.Br(),
                html.Br(),
                html.B("3. How has regional participation evolved over the last decade?"),
                html.P("This line chart tracks user registration trends across top countries, revealing shifting momentum in global engagement with the data science community."),
                html.Iframe(
                    srcDoc= render_map_from_df(df= growth_map, GrowthMap= True),
                    style={"width": "100%", "height" : "600px", "border": "none"}
                ),
                html.Br(),
                html.Br(),
                html.H4("Concluding Stage 2: The Community's Climb", className="text-success mt-5"),
                html.P("Stage 2 concludes with a deep dive into the evolving Kaggle community — from the journeys of Grandmasters to the rise of beginners and the global spread of talent. This stage highlights the human side of competition, setting the stage for exploring how collaboration,and creativity shape Kaggle’s future.")
               
            ])
        ], className="container mb-4")
    
    elif clicked_id == "btn-stage3":

        stage3Fig1 = go.Figure()

        stage3Fig2 = make_subplots(
            rows= 1,
            cols= 3,
            horizontal_spacing=0.12,
            subplot_titles=(
                "Average Total Downloads per Dataset",
                "Average Total Votes per Dataset",
                "Average Total Views per Dataset"
            )
        )

        # === Create subplot grid: 2 rows, 3 columns ===
        stage3Fig3 = make_subplots(
            rows=2, cols=3,
               specs=[
                        [{"colspan": 2}, None, {}],     # 1st row: [Plot 1 spans col 1-2, Plot 2, empty]
                        [{}, {}, {}]
                ],       # No colspan
            subplot_titles=[
                "Kaggle Notebook Creation per Year (2017–2024)",
                "Accelerator Usage in Notebooks (2017–2024)",
                "Average TotalVotes per Notebook",
                "Average TotalComments per Notebook",
                "Average TotalViews per Notebook",
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        stage3Fig4 = make_subplots(
            rows= 1,
            cols= 3,
            subplot_titles= [
                "Practice Competitions Over Time",
                "Unique Kagglers in Practice Competitions",
                "Prized Competitions Over Time"
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        # STAGE 3 FIGURE 1 PLOTTING
        stage3Fig1.add_trace(go.Scatter(
            x = yearly_dataset_count['Year Created'], y = yearly_dataset_count['count'],
            mode = "lines+markers",
            name = "DateSet Count"
        ))
        stage3Fig1.update_layout(
          title = "Kaggle Datasets Creation per Year (2016–2024)",
          xaxis_title = "Year",
          yaxis_title = "Number of Datasets" ,
          height=500,
          margin=dict(t=40, b=40) 
        )

        # STAGE 3 FIGURE 2 PLOTTING
        stage3Fig2.add_trace(go.Scatter(
            x = engagement_dataset_avg["Year Created"], y = engagement_dataset_avg["TotalDownloads"],
            mode = "lines+markers",
            name = "Number of Downloads"
        ), row=1,col=1)

        stage3Fig2.add_trace(go.Scatter(
            x = engagement_dataset_avg["Year Created"], y = engagement_dataset_avg["TotalVotes"],
            mode = "lines+markers",
            name = "Number of Votes"
        ), row=1,col=2)

        stage3Fig2.add_trace(go.Scatter(
            x = engagement_dataset_avg["Year Created"], y = engagement_dataset_avg["TotalViews"],
            mode = "lines+markers",
            name = "Number of Views"
        ), row=1,col=3)
        
        # UPDATTING AXIS
        stage3Fig2.update_xaxes( title_text = "Year", row= 1, col= 1)
        stage3Fig2.update_xaxes( title_text = "Year", row= 1, col= 2)
        stage3Fig2.update_xaxes( title_text = "Year", row= 1, col= 3)
        stage3Fig2.update_yaxes( title_text = "Avg Total Downloads", row= 1, col= 1)
        stage3Fig2.update_yaxes( title_text = "Avg Total Votes", row= 1, col= 2)
        stage3Fig2.update_yaxes( title_text = "Avg Total Views", row= 1, col= 3)


        

        # STAGE 3 FIGURE 3 PLOTTING
        stage3Fig3.add_trace(
            go.Scatter(
                x=yearly_kernel_count["Year Created"],
                y=yearly_kernel_count["count"],
                mode="lines+markers",
                line=dict(color="darkgreen", width=3),
                name="Notebooks"
            ),
            row=1, col=1
        )

        #  Accelerator Usage
        for accel in ['GPU', 'TPU', 'None']:
            stage3Fig3.add_trace(
                go.Scatter(
                    x=percent["Year"],
                    y=percent[accel],
                    mode="lines+markers",
                    name=accel
                ),
                row=1, col=3
            )

        # Engagement Metrics
        metrics = ["TotalVotes", "TotalComments", "TotalViews"]
        for i, metric in enumerate(metrics):
            row = 2
            col = i + 1
            stage3Fig3.add_trace(
                go.Scatter(
                    x=engagement_kernel_avg["Year Created"],
                    y=engagement_kernel_avg[metric],
                    mode="lines+markers",
                    name=metric
                ),
                row=row, col=col
            )

        stage3Fig3.update_layout(
            height=800,
            width=1200,
            title_text="The Power of Public Code on Kaggle (2017–2024)",
            showlegend=True,
            template="plotly_white"
        )

        # Axis Labels 
        stage3Fig3.update_xaxes(title_text="Year", row=1, col=1)
        stage3Fig3.update_yaxes(title_text="Number of Notebooks", row=1, col=1)

        stage3Fig3.update_xaxes(title_text="Year", row=1, col=3)
        stage3Fig3.update_yaxes(title_text="Usage (%)", row=1, col=3)

        stage3Fig3.update_xaxes(title_text="Year", row=2, col=1)
        stage3Fig3.update_yaxes(title_text="Avg Votes", row=2, col=1)

        stage3Fig3.update_xaxes(title_text="Year", row=2, col=2)
        stage3Fig3.update_yaxes(title_text="Avg Comments", row=2, col=2)

        stage3Fig3.update_xaxes(title_text="Year", row=2, col=3)
        stage3Fig3.update_yaxes(title_text="Avg Views", row=2, col=3)


        # STAGE 3 FIGURE 4 PLOTTING

        stage3Fig4.add_trace(go.Bar(
            x= yearly_practice_counts['YearHosted'], y= yearly_practice_counts['count'],
            name="Practice Competitons",
            marker_color="#15999b",
            showlegend= True
        ), row= 1, col= 1)

        stage3Fig4.add_trace(go.Scatter(
            x = user_practice_per_year["YearHosted"], y = user_practice_per_year["UniqueUsers"],
            mode = "lines+markers",
            name = "Number of unique kagglers"
        ), row=1, col= 2)
         
        
        stage3Fig4.add_trace(go.Scatter(
            x=df_ai_invest['Year'], y=df_ai_invest['Sum_of_deals'],
            mode='lines+markers', name="AI Investment", line=dict(color='green')
        ), row=1, col=3)
        
        
        stage3Fig4.update_layout(
            height=400,
            width=1300,
            title_text="Kaggle's Practice Competitions — Building a Training Ground(2014-2024)",
            showlegend=True,
            template="plotly_white"
        )
        stage3Fig4.update_xaxes(title_text="Year", row= 1, col=1)
        stage3Fig4.update_yaxes(title_text="Number of Practice Competitions", row=1, col=1)
        stage3Fig4.update_xaxes(title_text="Year", row=1, col=2)
        stage3Fig4.update_yaxes(title_text="Number of Unique Kagglers", row=1, col=2)
        stage3Fig4.update_xaxes(title_text="Year", row=1, col=3)
        stage3Fig4.update_yaxes(title_text="Sum of Deals", row=1, col=3)

        return html.Div([
            html.H4("Stage 3: Kaggle’s Mark on the AI World", className="text-primary mt-4"),

            html.P("In this stage, we flip the lens: rather than exploring how AI or the community has shaped Kaggle, "
                    "we ask how Kaggle itself has left its imprint on the broader AI ecosystem. From powering open research "
                    "to developing a new generation of practitioners, this stage highlights Kaggle’s role in shaping real-world AI innovation."),

            html.H5("Insights Covered in Stage 3", className="text-primary mt-4"),
            html.Ul([
                html.Li("📊 The Rise of Datasets — How Kaggle became a hub for data sharing and benchmarking."),
                html.Li("💡 The Power of Public Code — Exploring the value of Kaggle Notebook and its GPU's"),
                html.Li("🧪 Practice Competitions — How Kaggle turned competition into a learning platform."),
                html.H5("The Rise of Datasets", className="text-primary mt-4"),
                html.P("How did datasets transform Kaggle into the world’s premier data playground — and what trends have driven its rise? This visualization tracks the surge in public datasets, highlighting shifts in topics, contributor growth, and benchmarking impact over time."),
                html.Br(),
                dcc.Graph(figure= stage3Fig1),
                dcc.Graph(figure= stage3Fig2),
                html.H5("The Power of Public Code", className="text-primary mt-4"),
                html.P("What makes public code on Kaggle so powerful — and how have GPUs supercharged that impact? This visualization delves into notebook creation trends, accelerator adoption, and engagement metrics to reveal the growing influence of shared analysis in the data science community."),
                dcc.Graph(figure= stage3Fig3),
                html.H5("Practice Competitions", className="text-primary mt-4"),
                html.P("How have practice competitions reshaped learning on Kaggle — and what do participant progress patterns reveal about skill development? This visualization uncovers trends in competition participation, the pace of learner advancement, and milestone achievements as Kaggle evolved into an interactive training ground."),
                html.Br(),
                dcc.Graph(figure= stage3Fig4),
                html.Br(),
                html.Br(),
                html.H4("Concluding Stage 3: Kaggle’s Mark on the AI World", className="text-success mt-5"),
                html.P(
                    "As Kaggle continues to evolve, its influence on the AI landscape grows deeper — from shaping benchmark standards "
                    "to fueling real-world innovation. Stage 3 highlights how Kaggle has gone beyond being just a platform — it has become "
                    "part of the infrastructure of modern AI.",
                    className="mb-4"
                ),

                html.Hr(),  # Horizontal line separator

                # Final Dashboard Outro
                html.H4("Wrapping Up", className="text-primary mt-4"),
                html.P(
                    "This journey through the Meta Kaggle datasets offers more than insights — it’s a reflection of how community, "
                    "curiosity, and competition have driven the co-evolution of Kaggle and AI. As we look to the future, Kaggle remains "
                    "not only a proving ground for data scientists, but a vital pulse of the machine learning world.",
                    className="mb-5"
                ),
                html.Hr(),

                html.H5("About this Dashboard", className="text-primary mt-5"),
                
                html.P(
                    "This dashboard was created as part of the Kaggle Hackathon 2025 using Python, Dash, and Plotly. "
                    "The insights are powered by the Meta Kaggle dataset — processed using pandas "
                    "and visualized with interactive plots and custom layouts."
                ),

                html.P(
                    "Design choices were made to keep the experience clean, modular, and scroll-friendly. "
                    "Each stage represents a thematic deep dive into Kaggle’s evolution alongside the AI industry."
                ),

                html.P([
                    "Created by Kaustuk Pratap Singh",
                    html.Br(),
                    html.A("Kaggle Profile Link", href="https://www.kaggle.com/kaustukpratapsingh", target="_blank"),
                    html.Br(),
                    html.A("Linkedin Profile Link", href="https://www.linkedin.com/in/kaustuk-pratap-singh-23291b36b/", target="_blank")
                    
                ], className="mb-5")
            ])

        ], className= "container mb-4")
                

    return ""

server = app.server

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080, debug=True)

