import pandas as pd 
import seaborn as sb 
import streamlit as st
import numpy as np 
import json
import os
import csv
import calendar
import sys 
import time
import warnings 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import requests
import datetime
import subprocess
import africastalking
import yfinance as yf
import openai
import os
import pyttsx3 as pt
from gtts import gTTS

from datetime import datetime, timedelta
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from dateutil.relativedelta import relativedelta
from streamlit_chat import message
# from openai import OpenAI
# client = OpenAI()



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.ml_models import regression_model, future_prediction



africastalking.initialize(
    username='EMID',
    api_key='atsk_58f201e4dcc87d75abfc0204afda8e79bab0a63530fac99a1c8935000412d9362cb6dcf3'
)

sms = africastalking.SMS
airtime = africastalking.Airtime

sb.set()
sb.set_style('darkgrid')
sb.set_palette('viridis')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

warnings.filterwarnings('ignore')


try:
    # check if the key exists in session state
    _ = st.session_state.keep_graphics
except AttributeError:
    # otherwise set it to false
    st.session_state.keep_graphics = False


with st.sidebar:
	selected = option_menu(
		menu_title = 'PesaWise',
		options = ['Chat', 'Loans & Savings', 'Stocks', 'Live Trading', 'Model'],
		icons = ['speedometer', 'graph-up-arrow', ':moneybag:', ':money_with_wings:'],
		menu_icon = 'cast',
		default_index = 0
		)
def load_lottiefile(filepath: str):
	with open(filepath, 'r') as file:
		return json.load(file)

fin_lottie = load_lottiefile('D:/Web_Development/Streamlit Deployment Projects/Dreamers Vault/animation/fin1.json')
fin2_lottie = load_lottiefile('D:/Web_Development/Streamlit Deployment Projects/Dreamers Vault/animation/fin2.json')
fin3_lottie = load_lottiefile('D:/Web_Development/Streamlit Deployment Projects/Dreamers Vault/animation/fin3.json')


if selected == 'Chat':
	
	st.image('D:/Web_Development/Streamlit Deployment Projects/Chatbot/chat1.png', width = 400)
	st.header('Chat GPT-3 Bot API')
	st.subheader('Chatbot')
	st.markdown("""

	## What is a chatbot?
	A chatbot is software that simulates human-like conversations with users via chat. Its key task is to answer user questions with instant messages

	Building my first AI chatbots with ready-to-use assistance. 
	I will be customizing them to fit my interest needs, and bring chatbots to life within minutes.

	### **Engage**
	Reach out to users proactively using personalized chatbot greetings. Turn online visits into social opportunities.

	### **Nurture**
	Lead users to a conversation through recommended chat and tailored communications.

	### **Qualify**
	Generate and qualify prospects automatically. Transfer high-intent leads to your sales reps in real time to shorten the sales cycle.

	### **Convert**
	Let customers purchase, order, or schedule meetings easily using a smart chatbot.
		
	### 
	Using [Openai platform](https://platform.openai.com/account/api-keys) chatgpt-3 to generate api and pull Get request from their servers.
	""")

	with st.form(key = 'user_submit'):
		ai_bot = st.write('AI: How may I help you', '''
			''')

		human = st.text_area('Human: ', '''
			''')

		submit = st.form_submit_button(label = 'submit')

	audio = pt.init()
	audio.say(ai_bot)

	openai.api_key = "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"

	start_sequence = ai_bot
	restart_sequence = human


	response = openai.Completion.create(
	  model="text-davinci-003",
	  prompt=str(human),
	  temperature=0.9,
	  max_tokens=150,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0.6,
	  stop=[" Human:", " AI:"]
	)

	print(completion.choices[0].message)


	if submit==True:
	    ai_bot_resp = st.write(response.choices[0].text)
	else:
	    st.write('How may I help you freind')

	while True:
		audio = pt.init()
		audio.say('Here is the answer to your question')
		audio.say(response.choices[0].text)
		audio.runAndWait()



if selected == 'Loans & Savings':
	st.write('''
		# Finance
		Finance is the study and discipline of [money](https://en.wikipedia.org/wiki/Money), [currency](https://en.wikipedia.org/wiki/Currency) and [capital assets](https://en.wikipedia.org/wiki/Capital_asset). 
		It is related to, but not synonymous with [economics](https://en.wikipedia.org/wiki/Economics), which is the study of production, distribution, and consumption of money, assets, goods and services (the discipline of financial economics bridges the two). 
		Finance activities take place in financial systems at various scopes, thus the field can be roughly divided into [personal, corporate, and public finance].

		''')
	col1, col2 = st.columns(2)
	with col1:
		st_lottie(fin_lottie,
			speed=1,
			reverse=False,
			loop=True,
			quality="high",
			width=300,
			height=250,
			)
		st.write('''

			### Areas of Finance
			- **Personal Finance** -
			the mindful planning of monetary spending and saving, while also considering the possibility of future risk". Personal finance may involve paying for education, 
			financing durable goods such as real estate and cars, buying insurance, investing, and saving for retirement.

			- **Corporate Finance** -
			deals with the actions that managers take to increase the value of the firm to the shareholders, the sources of funding and the capital structure of corporations, 
			and the tools and analysis used to allocate financial resources.

			- **Public Finance** -
			describes finance as related to sovereign states, sub-national entities, and related public entities or agencies. It generally encompasses a long-term strategic 
			perspective regarding investment decisions that affect public entities.

			- **Quantitative Finance** -
			also referred to as "mathematical finance" – includes those finance activities where a sophisticated mathematical model is required.
			[read more](https://corporatefinanceinstitute.com/resources/data-science/quantitative-finance/#:~:text=Quantitative%20finance%20is%20the%20use,relates%20to%20portfolio%20management%20applications.)
			
			''')
	with col2:
		st.write('''

			### Financial system

			As above, the financial system consists of the flows of capital that take place between individuals and households [(personal finance)](https://en.wikipedia.org/wiki/Personal_finance), 
			governments [(public finance)](https://en.wikipedia.org/wiki/Public_finance), and businesses [(corporate finance)](https://en.wikipedia.org/wiki/Corporate_finance). "Finance" thus studies the process of channeling money from savers and 
			investors to entities that need it. [b] Savers and investors have money available which could earn interest or dividends if put to productive use. 
			Individuals, companies and governments must obtain money from some external source, such as loans or credit, when they lack sufficient funds to operate.
			
			''')
		st_lottie(fin2_lottie,
			speed=1,
			reverse=False,
			loop=True,
			quality="high",
			width=300,
			height=250,
			)
		st_lottie(fin3_lottie,
			speed=1,
			reverse=False,
			loop=True,
			quality="high",
			width=300,
			height=250,
			)
		st.write('**Money-festing** :smiley:')
	st.image('D:/Web_Development/Streamlit Deployment Projects/Dreamers Vault/source/Interest Payout.png')
	st.write('''
			### Loan Calculator

			This is app is going to determine how long it takes to repay a loan borrowed, given amount borrowed, interest rates and payment terms 
		''')
	
	
	with st.form(key='form1'):
		col1, col2, col3 = st.columns(3)
		with col1:
			loan_amount = st.number_input('amount borrowed', value=0, min_value=0, max_value=int(10e10))

		with col2:
			payment_rate = st.slider('Interest rate', 0.0, 10.0)/100.0

		with col3:
			monthly_amount = st.number_input('Monthly re-payment', min_value=0, max_value=int(10e10))

		# submit_label = f'<i class="fas fa-calculator">Calculate</i> '
		# submit = st.form_submit_button(submit_label)

		submit = st.form_submit_button(label='Calculate')

		#Determine the total period it takes to repay off a loan
		# bal = 5000
		# interestRate = 0.13
		# monthlyPayment = 500

	if submit not in st.session_state:
		df = pd.DataFrame(columns=['End Month', 'Loan Amount', 'Interest Charge'])
		
		current_date = datetime.today()
		# print(current_date)

		end_month_day = calendar.monthrange(current_date.year, current_date.month)[1]
		days_left = end_month_day - current_date.day

		next_month_start_date = current_date + timedelta(days=days_left + 1)
		end_month = next_month_start_date

		period_count = 0
		total_int = 0
		data = []

		while loan_amount > 0:
		    int_charge = (payment_rate / 12) * loan_amount
		    loan_amount += int_charge
		    loan_amount -= monthly_amount

		    if loan_amount <= 0:
		        loan_amount = 0
		    total_int += int_charge
		    print(end_month, round(loan_amount, 2), round(int_charge, 2))

		    period_count += 1
		    new_date = calendar.monthrange(end_month.year, end_month.month)[1]
		    end_month += timedelta(days=new_date)

		    # df = df.append({'End Month': end_month, 'Loan Amount': round(loan_amount, 2), 'Interest Charge': round(int_charge, 2)}, ignore_index=True)

		    data.append([end_month.date(), round(loan_amount, 2), round(int_charge, 2)])

		    if loan_amount == 0:
		        break

		print('Total Interest Rate paid: ', total_int)
		df = pd.DataFrame(data, columns=['next_pay_date', 'amount_remaining', 'interest_amount'])

		
		years = int(period_count // 12)
		months_remaining = round(period_count % 12)
		print(f"{years} years and {months_remaining} months")

		col1, col2 = st.columns(2)
		with col1:
			st.dataframe(df, use_container_width=True)

		with col2:
			st.write('Loan payment due')
			col1, col2, col3 = st.columns(3)
			col1.metric("", str(years), " yrs")
			col2.metric("", str(months_remaining), " months")
			st.metric("Total Interest Paid", "sh. " + str(round(total_int)), "")
			# col2.metric("Wind", "9 mph", "-8%")
			# col3.metric("Humidity", "86%", "4%")

	with st.popover("Download Report"):

		with st.form(key="report"):
			phone_number = st.number_input('Phone Number', value=0, min_value=0, max_value=int(10e10))

			submit_report = st.form_submit_button("Send")

			def send_report():
				amount = "10"
				currency_code = "KES"


				recipients = [f"+254{str(phone_number)}"]
				airtime_rec = "+254" + str(phone_number)
				print(recipients)
				# print(phone_number)

				# Set your message
				message = f"Welcome to PesaWise \n Your account was succesful created";
				# Set your shortCode or senderId
				sender = "AFTKNG"
				try:
					# responses = airtime.send(phone_number=airtime_rec, amount=amount, currency_code=currency_code)
					response = sms.send(message, recipients, sender)
					
					print(response)
					# print(responses)
				except Exception as e:
					print(f'Houston, we have a problem: {e}')

		if submit_report not in st.session_state:
			send_report()

		
		else:
			pass

			# send_report()







# --------------------------------------End Finance Section ----------------------------------------------------- #



if selected == 'Stocks':

	header = st.container()
	local_data = st.container()

	with header:
		st.image('../source/img7.jpg')
		st.title('GLOBAL STOCKS MARKET DATA')
		st.write('**What do stocks mean?**')
		st.write('A stock represents a share in the ownership of a company, including a claim on the companys earnings and assets. As such, stockholders are partial owners of the company. When the value of the business rises or falls, so does the value of the stock.')


	with local_data:
		col1, col2 = st.columns(2)
		
		with col1:
			st.image('../source/img6.png', width=350)

		with col2:
			st.image('../source/img4.webp', width=350)

		
		st.write("<h3 style='text-align: center; color: white;'>One place for your portfolios, <br>metrics and more<h3>", unsafe_allow_html=True)
		st.write('Gain insights, see trends and get real-time updates from well researched and analyzed datasets.')
		st.write('However the developer will integrate the results with Machine Learning algorithms for effecient and predictive output. This will boost accuracy and confidence in investing in stocks.')
		st.write('sorry I wasnt listening.....I was thinking about TRADING')


		df = pd.read_csv('../source/big_tech_stock_prices.csv')
		st.dataframe(df.head())

		# Data Cleaning
		df['date'] = pd.to_datetime(df['date'])

		st.write('### Statistical Representation of Data')

		col1, col2 = st.columns(2)
		
		with col1:
			st.write('Rows ', df.shape[0], 'Columns / Series ', df.shape[1])
			# st.write('Columns / Series ', df.shape[1])
		
			# Capture the output of df.info()
			buffer = io.StringIO()
			df.info(buf=buffer)
			info_str = buffer.getvalue()

			# Display df.info() output in Streamlit
			st.write('Summary of the dataframe')
			st.text(info_str)

		with col2:
			st.write('')
			st.write('')
			st.write('')
			st.write('')
			st.write('Description of the dataframe')
			st.dataframe(df.describe())

		st.write('### Graphical Presentation of Data')

		def get_numerical(df):
			numerical_list = []
			categories = df.select_dtypes(include=['float64', 'int64'])
			for i in categories:
				numerical_list.append(i)
			print(numerical_list)
			return numerical_list

		plot_hist_column = st.selectbox('Select dataframe series', (i for i in get_numerical(df)))

		print(plot_hist_column)

		fig = px.histogram(df[plot_hist_column], 
				title = 'Stock Distribution Plot for ' +  str(plot_hist_column) + ' series'
			)
		
		print(df[plot_hist_column])
		st.plotly_chart(fig)
		
		def get_categories(df):
		    cat = []
		    categories = df.select_dtypes(include=['float64', 'int64'])
		    for i in categories:
		        cat.append(i)
		    print(cat)
		    # fig = sb.heatmap(df[cat].corr(), annot=True, linewidths=0.5)
		    fig = px.imshow(df[cat].corr(), text_auto=True, aspect='auto', 
		    	title = 'Pearsons Correlation of Columns'
		    	)
		    st.plotly_chart(fig)

		get_categories(df)

		col1, col2 = st.columns(2)

		with col1:
			stock_company = st.selectbox('Select Symbol company', df['stock_symbol'].unique())

		with col2:
			stock_clause = st.selectbox('Select Stock Clause', get_numerical(df))

		company_group = df.groupby('stock_symbol').get_group(stock_company)
		
		
		pivot_df = company_group.pivot(index='stock_symbol', columns='date', values=stock_clause)
		print(pivot_df)

		fig = go.Figure(data=go.Heatmap(
			z=pivot_df,
	        x=company_group['date'],
	        y=company_group[stock_clause],
	        colorscale='Viridis'))
		
		fig.update_layout(
			title=f"Daily stocks charts from  {df['date'].dt.date.min()} to {df['date'].dt.date.max()}",
			# xaxis_title='Date',
			yaxis_title=f'{stock_clause} Price',
			legend_title='Company'
			)
		st.plotly_chart(fig)
		

		
		stock_symbol = df.groupby('stock_symbol').get_group(stock_company)

		
		fig = go.Figure()

		# Adding a trace for the company's open stock prices
		fig.add_trace(go.Scatter(x=stock_symbol['date'], y=stock_symbol['open'], mode='lines'))

		# frames = [go.Frame(data=[go.Scatter(x=company_group['date'][:k+1], y=company_group['open'][:k+1])],
	    #                name=str(company_group['date'].iloc[k])) for k in range(len(company_group.head(50)))]

		# fig.frames = frames

		# Setting the title and labels
		fig.update_layout(
			title=f'Open Stocks of the Tech company {stock_company}',
			xaxis_title='Date',
			yaxis_title='Open Price',
			legend_title='Company'
			)



		# Display the figure
		st.plotly_chart(fig)


		def load_lottiefile(filepath: str):
			with open(filepath, 'r') as file:
				return json.load(file)
		
		animation_1 = load_lottiefile('../source/stocks1.json')

		st_lottie(animation_1,
				speed=1,
				reverse=False,
				loop=True,
				quality="high",
				width=500,
				height=450,
				)
		
		# finance_lottie = load_lottieurl("https://app.lottiefiles.com/share/9e58a2cc-e627-4b6a-a0b4-3fa95571236c")


#############################################################################################################################

if selected == 'Live Trading':

# 	st.video('../source/stock_animation.mp4', format='mp4')

# 	import yfinance as yf

# 	tickers = yf.Tickers('msft aapl goog tsla scom coop kcb eqt kq nse bat bamb totl nmg nbk dtk')

# 	# access each ticker 
# 	stock = tickers.tickers[str('nmg').upper()].history(period="max")

# 	df2 = pd.DataFrame(stock).head(1000)

# 	if {'Dividends', 'Stock Splits'}.issubset(df2.columns):
# 		df2.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
	

# 	st.dataframe(df2)
# 	st.write(df2.shape)
# 	st.write(df2.columns)
# 	df2.reset_index(inplace=True)

# 	buffer = io.StringIO()
# 	df2.info(buf=buffer)
# 	info_str = buffer.getvalue()

# 	# Display df.info() output in Streamlit
# 	st.write('Summary of the dataframe')
# 	st.text(info_str)



# 	fig = go.Figure()

# 	fig.add_trace(go.Scatter(x=df2.index, y = df2['High'], mode='lines'))



# 	# Add frames for animation
# 	frames = [go.Frame(data=[go.Scatter(x=df2['Date'][:k+1], y=df2['High'][:k+1])],
# 	                   name=str(df2['Date'].iloc[k])) for k in range(len(df2))]

# 	fig.frames = frames

# 	# Update layout with animation settings
# 	fig.update_layout(
# 	    title=f'High Stocks of the Tech company',
# 	    xaxis_title='Date',
# 	    yaxis_title='High Price',
# 	    legend_title='Company'
# 	    # updatemenus=[dict(type='buttons', showactive=False,
# 	    #                   buttons=[dict(label='Play',
# 	    #                                 method='animate',
# 	    #                                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])],
# 	    # # Automatically start the animation
# 	    # transition={'duration': 100},
# 	    # # frame={'duration': 100, 'redraw': True},
# 	    # sliders=[dict(steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))], label=f.name) for f in frames])]
# 	)


# 	st.plotly_chart(fig)


	
##########################################################################################
	

	# Streamlit app
	st.title('Market Security Stocks')

	# User selects a stock symbol
	# stock_symbol = st.selectbox('Select Symbol company', ['AAPL', 'GOOG', 'MSFT'])

	
	with st.form(key="input_parameters"):

		tk = yf.Tickers('msft aapl goog tsla scom coop kcb eqt kq nse bat bamb totl nmg nbk dtk')



		symbol = []
		for i in tk.symbols:
			symbol.append(i)
			symbol.sort(reverse=False)


		ticker = st.selectbox('select ticker symbol', symbol)

		submitted = st.form_submit_button('explore')


		# Fetch the real-time data
		stock = tk.tickers[str(ticker).upper()].history(period="max")

		data = pd.DataFrame(stock)#.head(1000)

		st.write(data.index.max())


		if {'Dividends', 'Stock Splits'}.issubset(data.columns):
			data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

		today_high = round(data["High"].iloc[0] - data["High"].iloc[1], 2)
		today_open = round(data["Open"].iloc[0] - data["Open"].iloc[1], 2)
		today_high = round(data["High"].iloc[0] - data["High"].iloc[1], 2)

		st.write(f'Market summary > {ticker}')

		trade_col1, trade_col2, trade_col3, trade_col4 = st.columns(4)

		with trade_col1:
			st.metric(label='Net Gain/Loss', value=str(round(data["High"].iloc[0], 2)), delta=str(today_high) + " Today")

		with trade_col2:
			st.metric(label='Open', value=str(round(data["Open"].iloc[0], 2)), delta=str(today_open))

		with trade_col3:
			st.metric(label='Date', value=str(datetime.today().year), delta=str(datetime.today().strftime('%A')))
			# st.write(str(datetime.today().date()))

	
		data.reset_index(inplace=True)

		# Placeholder for the plot
		placeholder = st.empty()


	if submitted or st.session_state.keep_graphics:

		future_prediction(data)
		regression_model(data)

		# Infinite loop to update the plot with real-time data
		while True:
		    # Create the plot
		    fig1 = go.Figure()
		    
		    fig1.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines'))
		    
		    # Update layout
		    fig1.update_layout(
		        title=f'{ticker.upper()}',
		        xaxis_title='Time',
		        yaxis_title='Price',
		        legend_title='Stock Symbol'
		    )
		    
		    # Update the plot in the placeholder
		    placeholder.plotly_chart(fig1, use_container_width=True, key="iris")
		    
		    # Wait for a few seconds before updating
		    time.sleep(5)  # Adjust the sleep time as needed

		st.plotly_chart(placeholder)

	
	




if selected ==  'Model':
	pass
	# regression_model(data)
	# subprocess.run([f"{sys.executable}", "../model/regression.py"])
	# st.plotly_chart(fig)

			