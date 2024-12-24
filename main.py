cols=['timestamp','bid_price','bid_size','ask_price','ask_size','trade_price','trade_size',
'volume','notional_volume','vwap','nbb_agg_size',
'nbb_no_of_exchange', 'nbo_agg_size']
cols_price=['timestamp','bid_price','bid_size','ask_price','ask_size','vwap','nbb_agg_size'
,'nbb_no_of_exchange', 'nbo_agg_size']
#Reading file
df=pd.read_csv('/content/drive/MyDrive/2021 Quant Programming Task/AMZN.csv')
#Getting rid of preceding whitespace
df.columns=[i.lstrip() for i in df.columns]
#Getting epoch at 9:30,4PM EST
min_tick_time=int(df['timestamp'].min()/1000)*1000
max_tick_time=int(df['timestamp'].max()/1000)*1000
stocks=['AMZN','BLDE','CAL','KXIN','PRTS','SHOO','SMTC','TECH','TTGT','XOS']

#Function to read csv
def read_data(stock):
df=pd.read_csv('/content/drive/MyDrive/2021 Quant Programming Task/'+stock+".csv")
df.columns=[i.lstrip() for i in df.columns]
return df
#Function to extract mkt info, reindex to regular interval and ffill
def pre_process_mkt(df,step=25):
df=df[df['timestamp']<=max_tick_time]
df=df[df['timestamp']>=min_tick_time]
mkt=df[cols_price]

trades=df[['timestamp','trade_price','trade_size','volume','notional_volume','is_bu
y_order']]
#Defining a date range index on intervals I want
date_index=pd.RangeIndex(
start=(min_tick_time),
stop=(max_tick_time),
step=step, name='timestamp').tolist()
mkt=mkt.drop_duplicates(keep='last')

mkt=mkt.set_index(['timestamp'])
mkt=mkt[~mkt.index.duplicated(keep='last')]
mkt=mkt.reindex(mkt.index.union(date_index))
mkt = mkt.fillna(method='ffill')
mkt = mkt.reindex(date_index).reset_index()
mkt=mkt.drop_duplicates(keep='last')# taking last market price as current
mkt.dropna(inplace=True)
mkt=mkt.reset_index()
mkt['timestamp']=pd.to_datetime(mkt['timestamp'],unit='ms')
mkt['timestamp']=mkt['timestamp']-timedelta(hours=4)
mkt['spread']=mkt['ask_price']-mkt['bid_price']
mkt=mkt.set_index(['timestamp'])
return mkt

# Trades related info
def get_trades(df):
df=df[df['timestamp']<=max_tick_time]
df=df[df['timestamp']>=min_tick_time]
trades=df[['timestamp','trade_price','trade_size','volume','notional_volume','is_bu
y_order']]

tradess=trades[['timestamp','trade_price','trade_size','volume','notional_volume','
is_buy_order']]
tradess=tradess.drop_duplicates()
#tradess['timestamp']=(tradess['timestamp']/1000).astype('int')
tradess['timestamp']=pd.to_datetime(tradess['timestamp'],unit='ms')
tradess['timestamp']=tradess['timestamp']-timedelta(hours=4)
tradess=tradess.set_index(['timestamp'])
return tradess
#for binning the data in intervals
def bin_mkt(df,interval='15T'):
dff1=df.groupby(pd.Grouper(freq=interval)).agg({'ask_price':'mean','bid_price':'mean','spre
ad':'mean','vwap':'mean'})
return dff1
def bin_trades(df,interval='15T'):
trade=df.groupby(pd.Grouper(freq=interval)).agg({'trade_price':'mean','trade_size':
'sum','volume':'max','trade_size':'sum'})
tot_vol=trade['trade_size'].sum()
trade['%_size']=trade['trade_size']/tot_vol
trade['%_size']=trade['%_size']*100
return trade

i=0
fig, ax1 = plt.subplots(5,2,figsize=(16,28))
for stock in stocks:
i1=i//2
i2=i%2
data=read_data(stock)
data1=pre_process_mkt(data)
data2=get_trades(data)
data11=bin_mkt(data1)
data22=bin_trades(data2)
data=data11.join(data22)

ax1[i1][i2].plot(data['vwap'],color="green")
ax2 = ax1[i1][i2].twinx()
ax2.bar(data.index.tolist(),data['%_size'],color="grey",width=0.005)
ax2.set_ylim(0,100)
ax3 = ax1[i1][i2].twinx()
ax3.plot(data['spread'],color="maroon")
#ax3.spines['right'].set_position(('outward',60))
ax2.spines['right'].set_position(('axes',1.15))
ax1[i1][i2].set_ylabel("VWAP",color="green")
ax2.set_ylabel("% volume",color="grey")
ax3.set_ylabel("Spread",color="maroon")
ax1[i1][i2].tick_params(axis='y',colors="green")
ax2.tick_params(axis='y',colors="grey")
ax3.tick_params(axis='y',colors="maroon")
ax2.spines['right'].set_color("grey")
ax3.spines['right'].set_color("maroon")
ax1[i1][i2].spines['left'].set_color("green")
plt.title(stock)
i+=1
fig.tight_layout()

def reindex_newdates(mkt, new_dates):
"""
Fill the required times (as per the trades) in the dataframe

and forward fill the data. Returns updated dataframe.
"""
#mkt['timestamp']=pd.to_datetime(mkt['timestamp'],unit='ms')
new_index = mkt["timestamp"].append(
new_dates).sort_values().drop_duplicates()
mkt = mkt.set_index('timestamp')
mkt = mkt[~mkt.index.duplicated(keep='last')]
#print('after set last')
mkt = mkt.reindex(new_index, method='ffill')
#print('after fill')
#print(tick_df)
mkt.index.name = 'timestamp'
return mkt.reset_index()
def find_timed_ticks(mkt, trades_df):
"""
Mapping the mkid to the time required . Returns the mapped trades Data.
"""
time_df = pd.DataFrame(trades_df['timestamp'])
time_df['diff']="tradetime"
temp_df = pd.DataFrame()
temp_df['timestamp'] =trades_df['timestamp'] + pd.Timedelta('10s')
temp_df['diff']="10s"
time_df = time_df.append(temp_df)
mkt = reindex_newdates(mkt, time_df['timestamp'])
mkt = mkt.set_index('timestamp')
time_df.index.name = 'temp_id'
time_df = time_df.reset_index().set_index('timestamp')
merged_df = time_df.join(mkt)
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.set_index(['temp_id','diff'])
merged_df = merged_df.unstack(level=-1)
merged_df.columns = ["10s","tradetime"]
merged_df = trades_df.join(merged_df)
return merged_df

dat_f=pd.DataFrame()
for stock in stocks:
print(stock)
data=read_data(stock)
mkt=pre_process_mkt(data)

tradess=get_trades(data)
tradess=tradess.reset_index()
mkt['mid_price']=(mkt['ask_price']+mkt['bid_price'])/2
mkt=mkt.reset_index()
mkt_trades_df=find_timed_ticks(mkt[['timestamp','mid_price']],tradess[['timestamp']])
mkt_trades_df.dropna(inplace=True)
data['timestamp']=pd.to_datetime(data['timestamp'],unit='ms')
data['timestamp']=data['timestamp']-timedelta(hours=4)
dat=pd.merge(data,mkt_trades_df,on='timestamp')
dat=dat.drop_duplicates(keep='last')

dat['norm_mid']=((dat["10s"]-dat["tradetime"])/dat["tradetime"])/(dat['ask_price']-dat['bid
_price'])
dat['bo_imbalance']=(dat['nbo_agg_size']-dat['nbb_agg_size'])/(dat['nbo_agg_size']+dat['nbb
_agg_size'])
dat['avg_bid_ask']=(dat['ask_price']+dat['bid_price'])/2
dat['trade_sign']=-1
dat.loc[dat['trade_price']>=dat['avg_bid_ask'],'trade_sign']=1
dat=dat.loc[(dat['ask_price']-dat['bid_price'])>0]
dat['trade_imbalance']=dat['trade_sign']*dat['trade_size']
dat1=dat.set_index('timestamp')
dat1['trade_imb_last10']=dat1['trade_imbalance'].rolling('10s').sum()
dat1=dat1.reset_index(drop=True)
dat1=dat1[['bo_imbalance','trade_sign','trade_imb_last10','norm_mid']]
dat1.dropna(inplace=True)
dat_f=dat_f.append(dat1)
from sklearn.linear_model import LinearRegression

X=dat_f.iloc[:,:-1]
y=dat_f.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regr=LinearRegression()
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)
print(mean_absolute_error(y_test,y_pred))
print(sum(y_test)/len(y_test))
print(np.median(y_pred))
print(np.median(y_test))
v=pd.DataFrame(X_test)
v['pred']=y_pred
v['test']=y_test
