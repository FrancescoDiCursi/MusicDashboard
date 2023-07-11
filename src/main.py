import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import os
import datetime
import numpy as np

from textblob import TextBlob
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont
import nltk
#nltk.download("stopwords") #this slows down the reneder instance,
#from nltk.corpus import stopwords

import re

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import flask
from tqdm import tqdm

# try with only en from https://github.com/stopwords-iso/stopwords-iso/blob/master/stopwords-iso.json
stopwords=["'ll","'tis","'twas","'ve","10","39","a","a's","able","ableabout","about","above","abroad","abst","accordance","according","accordingly","across","act","actually","ad","added","adj","adopted","ae","af","affected","affecting","affects","after","afterwards","ag","again","against","ago","ah","ahead","ai","ain't","aint","al","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","amoungst","amount","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","ao","apart","apparently","appear","appreciate","appropriate","approximately","aq","ar","are","area","areas","aren","aren't","arent","arise","around","arpa","as","aside","ask","asked","asking","asks","associated","at","au","auth","available","aw","away","awfully","az","b","ba","back","backed","backing","backs","backward","backwards","bb","bd","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","bf","bg","bh","bi","big","bill","billion","biol","bj","bm","bn","bo","both","bottom","br","brief","briefly","bs","bt","but","buy","bv","bw","by","bz","c","c'mon","c's","ca","call","came","can","can't","cannot","cant","caption","case","cases","cause","causes","cc","cd","certain","certainly","cf","cg","ch","changes","ci","ck","cl","clear","clearly","click","cm","cmon","cn","co","co.","com","come","comes","computer","con","concerning","consequently","consider","considering","contain","containing","contains","copy","corresponding","could","could've","couldn","couldn't","couldnt","course","cr","cry","cs","cu","currently","cv","cx","cy","cz","d","dare","daren't","darent","date","de","dear","definitely","describe","described","despite","detail","did","didn","didn't","didnt","differ","different","differently","directly","dj","dk","dm","do","does","doesn","doesn't","doesnt","doing","don","don't","done","dont","doubtful","down","downed","downing","downs","downwards","due","during","dz","e","each","early","ec","ed","edu","ee","effect","eg","eh","eight","eighty","either","eleven","else","elsewhere","empty","end","ended","ending","ends","enough","entirely","er","es","especially","et","et-al","etc","even","evenly","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","fairly","far","farther","felt","few","fewer","ff","fi","fifteen","fifth","fifty","fify","fill","find","finds","fire","first","five","fix","fj","fk","fm","fo","followed","following","follows","for","forever","former","formerly","forth","forty","forward","found","four","fr","free","from","front","full","fully","further","furthered","furthering","furthermore","furthers","fx","g","ga","gave","gb","gd","ge","general","generally","get","gets","getting","gf","gg","gh","gi","give","given","gives","giving","gl","gm","gmt","gn","go","goes","going","gone","good","goods","got","gotten","gov","gp","gq","gr","great","greater","greatest","greetings","group","grouped","grouping","groups","gs","gt","gu","gw","gy","h","had","hadn't","hadnt","half","happens","hardly","has","hasn","hasn't","hasnt","have","haven","haven't","havent","having","he","he'd","he'll","he's","hed","hell","hello","help","hence","her","here","here's","hereafter","hereby","herein","heres","hereupon","hers","herself","herse”","hes","hi","hid","high","higher","highest","him","himself","himse”","his","hither","hk","hm","hn","home","homepage","hopefully","how","how'd","how'll","how's","howbeit","however","hr","ht","htm","html","http","hu","hundred","i","i'd","i'll","i'm","i've","i.e.","id","ie","if","ignored","ii","il","ill","im","immediate","immediately","importance","important","in","inasmuch","inc","inc.","indeed","index","indicate","indicated","indicates","information","inner","inside","insofar","instead","int","interest","interested","interesting","interests","into","invention","inward","io","iq","ir","is","isn","isn't","isnt","it","it'd","it'll","it's","itd","itll","its","itself","itse”","ive","j","je","jm","jo","join","jp","just","k","ke","keep","keeps","kept","keys","kg","kh","ki","kind","km","kn","knew","know","known","knows","kp","kr","kw","ky","kz","l","la","large","largely","last","lately","later","latest","latter","latterly","lb","lc","least","length","less","lest","let","let's","lets","li","like","liked","likely","likewise","line","little","lk","ll","long","longer","longest","look","looking","looks","low","lower","lr","ls","lt","ltd","lu","lv","ly","m","ma","made","mainly","make","makes","making","man","many","may","maybe","mayn't","maynt","mc","md","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","mh","microsoft","might","might've","mightn't","mightnt","mil","mill","million","mine","minus","miss","mk","ml","mm","mn","mo","more","moreover","most","mostly","move","mp","mq","mr","mrs","ms","msie","mt","mu","much","mug","must","must've","mustn't","mustnt","mv","mw","mx","my","myself","myse”","mz","n","na","name","namely","nay","nc","nd","ne","near","nearly","necessarily","necessary","need","needed","needing","needn't","neednt","needs","neither","net","netscape","never","neverf","neverless","nevertheless","new","newer","newest","next","nf","ng","ni","nine","ninety","nl","no","no-one","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","notwithstanding","novel","now","nowhere","np","nr","nu","null","number","numbers","nz","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","om","omitted","on","once","one","one's","ones","only","onto","open","opened","opening","opens","opposite","or","ord","order","ordered","ordering","orders","org","other","others","otherwise","ought","oughtn't","oughtnt","our","ours","ourselves","out","outside","over","overall","owing","own","p","pa","page","pages","part","parted","particular","particularly","parting","parts","past","pe","per","perhaps","pf","pg","ph","pk","pl","place","placed","places","please","plus","pm","pmid","pn","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","pr","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provided","provides","pt","put","puts","pw","py","q","qa","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","reasonably","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","reserved","respectively","resulted","resulting","results","right","ring","ro","room","rooms","round","ru","run","rw","s","sa","said","same","saw","say","saying","says","sb","sc","sd","se","sec","second","secondly","seconds","section","see","seeing","seem","seemed","seeming","seems","seen","sees","self","selves","sensible","sent","serious","seriously","seven","seventy","several","sg","sh","shall","shan't","shant","she","she'd","she'll","she's","shed","shell","shes","should","should've","shouldn","shouldn't","shouldnt","show","showed","showing","shown","showns","shows","si","side","sides","significant","significantly","similar","similarly","since","sincere","site","six","sixty","sj","sk","sl","slightly","sm","small","smaller","smallest","sn","so","some","somebody","someday","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","sr","st","state","states","still","stop","strongly","su","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","sv","sy","system","sz","t","t's","take","taken","taking","tc","td","tell","ten","tends","test","text","tf","tg","th","than","thank","thanks","thanx","that","that'll","that's","that've","thatll","thats","thatve","the","their","theirs","them","themselves","then","thence","there","there'd","there'll","there're","there's","there've","thereafter","thereby","thered","therefore","therein","therell","thereof","therere","theres","thereto","thereupon","thereve","these","they","they'd","they'll","they're","they've","theyd","theyll","theyre","theyve","thick","thin","thing","things","think","thinks","third","thirty","this","thorough","thoroughly","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","till","tip","tis","tj","tk","tm","tn","to","today","together","too","took","top","toward","towards","tp","tr","tried","tries","trillion","truly","try","trying","ts","tt","turn","turned","turning","turns","tv","tw","twas","twelve","twenty","twice","two","tz","u","ua","ug","uk","um","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","upwards","us","use","used","useful","usefully","usefulness","uses","using","usually","uucp","uy","uz","v","va","value","various","vc","ve","versus","very","vg","vi","via","viz","vn","vol","vols","vs","vu","w","want","wanted","wanting","wants","was","wasn","wasn't","wasnt","way","ways","we","we'd","we'll","we're","we've","web","webpage","website","wed","welcome","well","wells","went","were","weren","weren't","werent","weve","wf","what","what'd","what'll","what's","what've","whatever","whatll","whats","whatve","when","when'd","when'll","when's","whence","whenever","where","where'd","where'll","where's","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","whichever","while","whilst","whim","whither","who","who'd","who'll","who's","whod","whoever","whole","wholl","whom","whomever","whos","whose","why","why'd","why'll","why's","widely","width","will","willing","wish","with","within","without","won","won't","wonder","wont","words","work","worked","working","works","world","would","would've","wouldn","wouldn't","wouldnt","ws","www","x","y","ye","year","years","yes","yet","you","you'd","you'll","you're","you've","youd","youll","young","younger","youngest","your","youre","yours","yourself","yourselves","youve","yt","yu","z","za","zero","zm","zr"]



app= dash.Dash(__name__)
server = app.server
#_____________________________________
#CLEAN DATA

df_l=[pd.read_csv(f"./Data/{x}") for x in os.listdir("./Data") if x.endswith(".csv")]
df=pd.concat(df_l)
df.reset_index(drop=True, inplace=True)
df["activity"]=[f"{x.split('–')[0]} - {x.split('–')[1].replace('in attività',str(datetime.datetime.now().year))}"\
                for x in df["activity"]]
df["song len"]=[int(x.split(":")[0])*60 + int(x.split(":")[1])\
                 if x!="0" else 0 for x in df["song len"]]

df["count"]=[1 for x in df["song len"]]

filtered_df=df #modfy on demand



#___________________________________________________
#FUNCTIONS

def create_gnatt_df_activity(df):
    df_gnatt = df[["band name","activity"]].drop_duplicates().reset_index(drop=True)

    end_act=[]
    for i,each in enumerate(df_gnatt["activity"]):
        if ";" in str(each):
                df_gnatt=pd.concat([df_gnatt,pd.DataFrame( [[df_gnatt["band name"][i], df_gnatt["activity"][i].split(";")[1]]], columns=["band name","activity"]) ]  )
                df_gnatt.reset_index(drop=True, inplace=True)


    for i,each in enumerate(df_gnatt["activity"]):
        if len(str(each).split(";")[0].split("-"))==2:
            end_act.append(pd.to_datetime(int(str(each).split(";")[0].split("-")[1]), format="%Y"))
        else:
            end_act.append(pd.to_datetime(datetime.datetime.now().year,format="%Y"))
        df_gnatt.loc[i,"activity"]= pd.to_datetime(int(str(each).split(";")[0].split("-")[0]), format="%Y")

    df_gnatt["end"]=end_act
    return df_gnatt

def create_scatter_dfg_activity(df):
    df_scatter=df.copy()
    df_scatter=df_scatter[df_scatter["album year"]>1900]
    df_scatter["album year"]=[pd.to_datetime(x,format="%Y") for x in df_scatter["album year"]]
    df_scatter["count"]=[1 for x in df_scatter["band name"]]
    df_scatter_g=df_scatter[["album year", "band name","album title","song len","count"]].groupby(by=["band name","album year","album title"], as_index=False).sum()
    return df_scatter_g


def create_gnatt_marginal_x(df):
    #print("MARGINAL DF:  ", df)
    df_scatter_x = create_scatter_dfg_activity(df)

    #print("marginal x", df_scatter_x)
    if df_scatter_x.empty:
        fig= go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers"))
        fig.update_layout(template="plotly_dark")
    else:
        fig= px.bar(df_scatter_x, x="album year", y="song len",color="band name", log_y=True, text="album title")
        fig.update_xaxes(visible=True)
        fig.update_layout(template="plotly_dark")
        fig.update_layout(showlegend=True, legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
        ))
        df_line=df_scatter_x.groupby(["album year"],as_index=False).sum().sort_values(by=["album year"])
        fig.add_traces(go.Scatter(x=df_line["album year"], y=df_line["song len"], mode="lines", line=dict(color="red"), opacity=0.5, name="Total"))



    return fig

def create_gnatt_marginal_y(df):
    df_scatter_x = create_scatter_dfg_activity(df)
    if df_scatter_x.empty:
        fig= go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers"))
        fig.update_layout(template="plotly_dark")
    else:
        fig= px.bar(df_scatter_x, x="song len", y="band name",color="band name", log_x=True, orientation="h", text="album title")
        fig.update_xaxes(visible=True)
        fig.update_layout(template="plotly_dark")
        fig.update_layout(showlegend=True, legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig

def create_gnatt(df):
    df_gnatt = create_gnatt_df_activity(df)
    df_scatter_g = create_scatter_dfg_activity(df)

    if df_gnatt.empty:
        all_fig=go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers"))
    else:
       #print(df_gnatt.columns, df_scatter_g.columns)
        fig = px.timeline(df_gnatt, x_start="activity", x_end="end", y="band name",)
        fig2_color="count"
        fig2 = px.scatter(df_scatter_g, x="album year", y="band name", color=fig2_color, size="song len",
                          hover_name="album title", opacity=0.6)
        fig2.layout.coloraxis.colorbar.title = fig2_color

        fig2.update_traces(marker=dict(line=dict(width=1,
                                                color="gray",
                                                 )),
                         )

        #fig2.update_layout(hovermode="x unified")

        all_fig = go.Figure(data=fig2.data + fig.data)
    all_fig.update_layout(template="plotly_dark")
    all_fig.update_xaxes(showgrid=True)
    all_fig.update_yaxes(showgrid=True)
    return all_fig

def create_heats(df):
    new_rows = []

    df_heat_band_genre = df.copy()
    ("BAND GENRE", df_heat_band_genre["genres"])

    for i, each in enumerate(df_heat_band_genre["genres"]):
        if ";" in each:
            for j, el in enumerate(each.split(";")):
                new_rows.append([df_heat_band_genre["band name"][i], el, df_heat_band_genre["count"][i]])
        else:
            new_rows.append([df_heat_band_genre["band name"][i], df_heat_band_genre["genres"][i], df_heat_band_genre["count"][i]])
    df_heat_band_genre = pd.DataFrame(new_rows, columns=["band name", "genre", "count"])
    df_heat_band_genre
    ###
    new_rows = []

    df_heat_country_genre = df.copy()
    for i, each in enumerate(df_heat_country_genre["genres"]):
        if ";" in each:
            for j, el in enumerate(each.split(";")):
                new_rows.append([df_heat_country_genre["country"][i], el, df_heat_country_genre["count"][i]])
        else:
            new_rows.append([df_heat_country_genre["country"][i], df_heat_country_genre["genres"][i],
                             df_heat_country_genre["count"][i]])
    df_heat_country_genre = pd.DataFrame(new_rows, columns=["country", "genre", "count"])


    genres = sorted(df_heat_country_genre["genre"].unique())

    df_heat_band_genre = df_heat_band_genre.groupby(["genre", "band name"], as_index=False).sum()
    df_heat_country_genre = df_heat_country_genre.groupby(["genre", "country"], as_index=False).sum()
    df_heat_band_genre_piv = pd.DataFrame()
    df_heat_country_genre_piv = pd.DataFrame()
    try:
        df_heat_band_genre_piv = pd.pivot_table(df_heat_band_genre, index="band name", columns="genre", values="count")
        df_heat_country_genre_piv = pd.pivot_table(df_heat_country_genre, index="country", columns="genre", values="count")
    except:
        df_heat_band_genre_piv = pd.DataFrame([],columns=df.columns)
        df_heat_coutnry_genre_piv = pd.DataFrame([],columns=df.columns)


    #plot heats
    fig_heat_band_genres = px.imshow(
        df_heat_band_genre_piv,
        labels=dict(x="Genre", y="Band", color="count")
    )
    # fig_heat_band_genres.update_xaxes(side="top")
    fig_heat_band_genres.update_coloraxes(showscale=False)

    fig_heat_country_genres = px.imshow(df_heat_country_genre_piv,
                                        labels=dict(x="Genre", y="Band", color="count"))
    # fig_heat_country_genres.update_xaxes(side="top")
    fig_heat_country_genres.update_coloraxes(showscale=False)

    figures = [fig_heat_band_genres, fig_heat_country_genres]
    fig_heats = make_subplots(rows=len(figures), cols=1, shared_xaxes=True)

    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            fig_heats.append_trace(figure["data"][trace], row=i + 1, col=1)

    fig_heats.update_layout(template="plotly_dark")
    return fig_heats

def create_single_stats(df):
    fig_bandname = px.pie(names=df["band name"].value_counts().index, values=df["band name"].value_counts().values,
                 hole=0.5, labels={'band name': 'band name'})
    fig_bandname.update_traces(textposition='inside', textinfo='percent+label')
    fig_bandname.update_layout(title_text="band name", title_x=0.50, title_y=0.50, showlegend=False,
                               margin=dict(l=10, r=10, t=10, b=10),template="plotly_dark")

    fig_country = px.pie(names=df["country"].value_counts().index, values=df["country"].value_counts().values,
                 hole=0.5, labels={'country': 'country'})
    fig_country.update_traces(textposition='inside', textinfo='percent+label')
    fig_country.update_layout(title_text="country", title_x=0.50, title_y=0.50, showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10),template="plotly_dark")

    genres_=pd.Series([y for x in df["genres"] for y in x.split(';')])
    print(genres_.value_counts())

    fig_genres = px.pie(names=genres_.value_counts().index, values=genres_.value_counts().values,
                 hole=0.5, labels={'genre': 'genre'})
    fig_genres.update_traces(textposition='inside', textinfo='percent+label')
    fig_genres.update_layout(title_text="genres", title_x=0.50, title_y=0.50, showlegend=False,
                             margin=dict(l=10, r=10, t=10, b=10),template="plotly_dark")

    fig_albumtitle = px.pie(names=df["album title"].value_counts().index, values=df["album title"].value_counts().values,
                          hole=0.5, labels={'album title': 'album title'})
    fig_albumtitle.update_traces(textposition='inside', textinfo='percent+label')
    fig_albumtitle.update_layout(title_text="album title", title_x=0.50, title_y=0.50, showlegend=False,
                               margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")

    fig_songtitle = px.pie(names=df["song title"].value_counts().index, values=df["song title"].value_counts().values,
                          hole=0.5, labels={'song title': 'song title'})
    fig_songtitle.update_traces(textposition='inside', textinfo='percent+label')
    fig_songtitle.update_layout(title_text="song title", title_x=0.50, title_y=0.50, showlegend=False,
                               margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")

    #possibly comment out fig_albumyear as it is already in marginal distributions
    fig_albumyear= px.histogram(df[df["album year"]>1990],"album year",
                          log_y=True)
    fig_albumyear.update_layout( showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")

    fig_songlen= px.histogram(df, "song len", log_y=True)
    fig_songlen.update_layout( showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")

    fig_albumtitlesent= px.histogram(df, "album title sentiment",range_x=[-1,1], log_y=True)
    fig_albumtitlesent.update_layout( showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")

    fig_songtitlesent= px.histogram(df, "song title sentiment", range_x=[-1,1], log_y=True)
    fig_songtitlesent.update_layout( showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")

    fig_lyricsent= px.histogram(df, "song lyric sentiment", range_x=[-1,1], log_y=True)
    fig_lyricsent.update_layout( showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")





    return [fig_bandname, fig_country, fig_genres, fig_albumtitle, fig_songtitle, fig_albumyear, fig_songlen,
            fig_albumtitlesent, fig_songtitlesent, fig_lyricsent]


def create_map(df):
    country_to_iso3 = """ABW	Aruba	AW	533
    AFG	Afghanistan	AF	004
    AGO	Angola	AO	024
    AIA	Anguilla	AI	660
    ALB	Albania	AL	008
    AND	Andorra	AD	020
    ARE	Emirati Arabi Uniti	AE	784
    ARG	Argentina	AR	032
    ARM	Armenia	AM	051
    ASM	Samoa Americane	AS	016
    ATA	Antartide	AQ	010
    ATF	Terre australi e antartiche francesi	TF	260
    ATG	Antigua e Barbuda	AG	028
    AUS	Australia	AU	036
    AUT	Austria	AT	040
    AZE	Azerbaigian	AZ	031
    BDI	Burundi	BI	108
    BEL	Belgio	BE	056
    BEN	Benin	BJ	204
    BES	Bonaire, Sint Eustatius e Saba	BQ	535
    BFA	Burkina Faso	BF	854
    BGD	Bangladesh	BD	050
    BGR	Bulgaria	BG	100
    BHR	Bahrein	BH	048
    BHS	Bahamas	BS	044
    BIH	Bosnia ed Erzegovina	BA	070
    BLM	Saint-Barthélemy	BL	652
    BLR	Bielorussia	BY	112
    BLZ	Belize	BZ	084
    BMU	Bermuda	BM	060
    BOL	Bolivia	BO	068
    BRA	Brasile	BR	076
    BRB	Barbados	BB	052
    BRN	Brunei	BN	096
    BTN	Bhutan	BT	064
    BWA	Botswana	BW	072
    CAF	Repubblica Centrafricana	CF	140
    CAN	Canada	CA	124
    CCK	Isole Cocos	CC	166
    CHE	Svizzera	CH	756
    CHL	Cile	CL	152
    CHN	Cina	CN	156
    CIV	Costa d'Avorio	CI	384
    CMR	Camerun	CM	120
    COD	Repubblica Democratica del Congo	CD	180
    COG	Repubblica del Congo	CG	178
    COK	Isole Cook	CK	184
    COL	Colombia	CO	170
    COM	Comore	KM	174
    CPV	Capo Verde	CV	132
    CRI	Costa Rica	CR	188
    CUB	Cuba	CU	192
    CUW	Curaçao	CW	531
    CXR	Isola di Natale	CX	162
    CYM	Isole Cayman	KY	136
    CYP	Cipro	CY	196
    CZE	Repubblica Ceca	CZ	203
    DEU	Germania	DE	276
    DJI	Gibuti	DJ	262
    DMA	Dominica	DM	212
    DNK	Danimarca	DK	208
    DOM	Repubblica Dominicana	DO	214
    DZA	Algeria	DZ	012
    ECU	Ecuador	EC	218
    EGY	Egitto	EG	818
    ERI	Eritrea	ER	232
    ESH	Sahara Occidentale	EH	732
    ESP	Spagna	ES	724
    EST	Estonia	EE	233
    ETH	Etiopia	ET	231
    FIN	Finlandia	FI	246
    FJI	Figi	FJ	242
    FLK	Isole Falkland	FK	238
    FRA	Francia	FR	250
    FRO	Fær Øer	FO	234
    FSM	Stati Federati di Micronesia	FM	583
    GAB	Gabon	GA	266
    GBR	Inghilterra	GB	826
    GEO	Georgia	GE	268
    GGY	Guernsey	GG	831
    GHA	Ghana	GH	288
    GIB	Gibilterra	GI	292
    GIN	Guinea	GN	324
    GLP	Guadalupa	GP	312
    GMB	Gambia	GM	270
    GNB	Guinea-Bissau	GW	624
    GNQ	Guinea Equatoriale	GQ	226
    GRC	Grecia	GR	300
    GRD	Grenada	GD	308
    GRL	Groenlandia	GL	304
    GTM	Guatemala	GT	320
    GUF	Guyana francese	GF	254
    GUM	Guam	GU	316
    GUY	Guyana	GY	328
    HKG	Hong Kong	HK	344
    HND	Honduras	HN	340
    HRV	
    Croazia
    HR	191
    HTI	Haiti	HT	332
    HUN	Ungheria	HU	348
    IDN	Indonesia	ID	360
    IMN	Isola di Man	IM	833
    IND	India	IN	356
    IOT	Territorio britannico dell'Oceano Indiano	IO	086
    IRN	Iran	IR	364
    IRQ	Iraq	IQ	368
    IRL	Irlanda	IE	372
    ISL	Islanda	IS	352
    ISR	Israele	IL	376
    ITA	Italia	IT	380
    JAM	Giamaica	JM	388
    JOR	Giordania	JO	400
    JPN	Giappone	JP	392
    JEY	Jersey	JE	832
    KAZ	Kazakistan	KZ	398
    KEN	Kenya	KE	404
    KGZ	Kirghizistan	KG	417
    KHM	Cambogia	KH	116
    KIR	Kiribati	KI	296
    KNA	Saint Kitts e Nevis	KN	659
    KOR	Corea del Sud	KR	410
    KWT	Kuwait	KW	414
    LAO	Laos	LA	418
    LBN	Libano	LB	422
    LBR	Liberia	LR	430
    LBY	Libia	LY	434
    LCA	Saint Lucia	LC	662
    LIE	Liechtenstein	LI	438
    LKA	Sri Lanka	LK	144
    LSO	Lesotho	LS	426
    LTU	Lituania	LT	440
    LUX	Lussemburgo	LU	442
    LVA	Lettonia	LV	428
    MAC	Macao	MO	446
    MAF	Saint-Martin (Antille francesa)	MF	534
    MAR	Marocco	MA	504
    MCO	Monaco	MC	492
    MDA	Moldavia	MD	498
    MDG	Madagascar	MG	450
    MDV	Maldive	MV	462
    MEX	Messico	MX	484
    MHL	Isole Marshall	MH	584
    MKD	Macedonia	MK	807
    MLI	Mali	ML	466
    MLT	Malta	MT	470
    MMR	Birmania o Myanmar	MM	104
    MNE	Montenegro	ME	499
    MNG	Mongolia	MN	496
    MNP	Isole Marianne Settentrionali	MP	580
    MOZ	Mozambico	MZ	508
    MRT	Mauritanie	MR	478
    MSR	Montserrat	MS	500
    MTQ	Martinica	MQ	474
    MUS	Mauritius	MU	480
    MWI	Malawi	MW	454
    MYS	Malaysia	MY	458
    MYT	Mayotte	YT	175
    NAM	Namibia	NA	516
    NCL	Nuova Caledonia	NC	540
    NER	Niger	NE	562
    NFK	Isola Norfolk	NF	574
    NGA	Nigeria	NG	566
    NIC	Nicaragua	NI	558
    NIU	Niue	NU	570
    NLD	Paesi Bassi	NL	528
    NOR	Norvegia	NO	578
    NPL	Nepal	NP	524
    NRU	Nauru	NR	520
    NZL	Nuova Zelanda	NZ	554
    OMN	Oman	OM	512
    PAN	Panama	PA	591
    PAK	Pakistan	PK	586
    PCN	Isole Pitcairn	PN	612
    PER	Perù	PE	604
    PHL	Filippine	PH	608
    PLW	Palau	PW	585
    PNG	Papua Nuova Guinea	PG	598
    POL	Polonia	PL	616
    PRI	Porto Rico	PR	630
    PRK	Corea del Nord	KP	408
    PRT	Portogallo	PT	620
    PRY	Paraguay	PY	600
    PSE	Palestina	PS	275
    PYF	Polinesia Francese	PF	258
    QAT	Qatar	QA	634
    REU	Riunione	RE	638
    ROU	Romania	RO	642
    RUS	Russia	RU	643
    RWA	Ruanda	RW	646
    SAU	Arabia Saudita	SA	682
    SDN	Sudan	SD	729
    SEN	Senegal	SN	686
    SGP	Singapore	SG	702
    SGS	Georgia del Sud e Isole Sandwich Australi	GS	239
    SHN	Sant'Elena, Ascensione e Tristan da Cunha	SH	654
    SLB	Isole Salomone	SB	090
    SLE	Sierra Leone	SL	694
    SLV	El Salvador	SV	222
    SMR	San Marino	SM	674
    SOM	Somalia	SO	706
    SPM	Saint-Pierre e Miquelon	PM	666
    SRB	Serbia	RS	688
    SSD	Sudan del Sud	SS	728
    STP	São Tomé e Príncipe	ST	678
    SUR	Suriname	SR	740
    SVK	Slovacchia	SK	703
    SVN	Slovenia	SI	705
    SWE	Svezia	SE	752
    SWZ	eSwatani	SZ	748
    SXM	Saint-Martin (Antille olandese)	SX	663
    SYC	Seychelles	SC	690
    SYR	Siria	SY	760
    TCA	Turks e Caicos	TC	796
    TCD	Ciad	TD	148
    TGO	Togo	TG	768
    THA	Thailandia	TH	764
    TJK	Tagikistan	TJ	762
    TKL	Tokelau	TK	772
    TKM	Turkmenistan	TM	795
    TLS	Timor Est	TL	626
    TON	Tonga	TO	776
    TTO	Trinidad e Tobago	TT	780
    TUN	Tunisia	TN	788
    TUR	Turchia	TR	792
    TUV	Tuvalu	TV	798
    TWN	Taiwan	TW	158
    TZA	Tanzania	TZ	834
    UGA	Uganda	UG	800
    UKR	Ucraina	UA	804
    URY	Uruguay	UY	858
    USA	Stati Uniti	US	840
    UZB	Uzbekistan	UZ	860
    VAT	Vaticano	VA	336
    VCT	Saint Vincent e Grenadine	VC	670
    VEN	Venezuela	VE	862
    VGB	Isole Vergini britanniche	VG	092
    VIR	Isole Vergini americane	VI	850
    VNM	Vietnam	VN	704
    VUT	Vanuatu	VU	548
    WLF	Wallis e Futuna	WF	876
    WSM	Samoa	WS	882
    XKX	Kosovo	XK	153
    YEM	Yemen	YE	887
    ZAF	Sudafrica	ZA	710
    ZMB	Zambia	ZM	894
    ZWE	Zimbabwe	ZW	716""".split("\n")

    country_to_iso3 = [x.split("\t")[:2] for x in country_to_iso3]
    country_to_iso3 = {x[1]: x[0] for x in country_to_iso3 if len(x) == 2}
    country_to_iso3
    df_chor = df.copy()[df["country"] != "multinazionale"]
    df_chor["country"] = [country_to_iso3[x.strip().replace("Regno Unito","Inghilterra")].strip() for x in df_chor["country"]]
    df_chor["count"] = [1 for x in enumerate(df_chor["country"])]
    df_chor_g = df_chor[["country", "count"]].groupby(["country"], as_index=False).sum()

    fig_chor_color="count"
    fig_chor = px.choropleth(data_frame=df_chor_g, locations=df_chor_g["country"], locationmode="ISO-3",
                             color=[np.log(int(x)) for x in df_chor_g[fig_chor_color]], hover_name=df_chor_g["country"],
                             hover_data={"count": True, "country": False},
                             projection="natural earth",
                             template="plotly_dark")
    try:
        fig_chor.update_coloraxes(cmin=0,
                                  cmid=sorted([np.log(x) for x in df_chor_g["count"]])[int(len(df_chor_g["count"]) / 3)],
                                  cmax=max([np.log(x) for x in df_chor_g["count"]]), colorbar_tickmode="array",
                                  colorbar_tickvals=[0,
                                                     sorted([np.log(x) for x in df_chor_g["count"]])[
                                                         int(len(df_chor_g["count"]) / 3)],
                                                     max([np.log(x) for x in df_chor_g["count"]])],
                                  colorbar_ticktext=[0,
                                                     sorted([x for x in df_chor_g["count"]])[
                                                         int(len(df_chor_g["count"]) / 3)],
                                                     max([x for x in df_chor_g["count"]])
                                                     ],
                                  title=fig_chor_color
                                  )
    except:
        pass
    return fig_chor


def process_text(df,type_): #original process_text moved to "Lyrics stats > scraping"; precomputed values to quicken the dashboard
    positive_words, negative_words, neutral_words , positive_sentences, negative_sentences, neutral_sentences = [],[],[],[],[],[]

    neutral_sentences=pd.Series([y for x in df["neutral sents"] for y in str(x).split("###") if y!="nan"]).value_counts()
    positive_sentences=pd.Series([y for x in df["positive sents"] for y in str(x).split("###") if y!="nan" ]).value_counts()
    negative_sentences=pd.Series([y for x in df["negative sents"] for y in str(x).split("###") if y!="nan"]).value_counts()

    neutral_words= pd.Series([y for x in df["neutral words"] for y in str(x).split("###") if y!="nan"]).value_counts()
    positive_words=pd.Series([y for x in df["positive words"] for y in str(x).split("###") if y!="nan"]).value_counts()
    negative_words=pd.Series([y for x in df["negative words"] for y in str(x).split("###") if y!="nan"]).value_counts()


    return [[positive_words,negative_words, neutral_words], [positive_sentences, negative_sentences, neutral_sentences]]

def create_neutral_word_cloud(df,filtered_data):
    data=filtered_data[0][2]
    #print("neutral words")
    return create_word_cloud(data,"v","words")

def create_positive_word_cloud(df,filtered_data):
    data=filtered_data[0][0]
    #print("positive words")
    return create_word_cloud(data,"h","words")

def create_negative_word_cloud(df,filtered_data):
    data=filtered_data[0][1]
    #print("negative words")
    return create_word_cloud(data,"h","words")

#
def create_neutral_sent_cloud(df,filtered_data):
    data= filtered_data[1][2]
    #print("neutral sentences")
    return create_word_cloud(data,"v", "sentences")

def create_positive_sent_cloud(df,filtered_data):
    data= filtered_data[1][0]
    #print("positive sentences")
    return create_word_cloud(data,"h", "sentences")

def create_negative_sent_cloud(df,filtered_data):
    data= filtered_data[1][1]
    #print("negative sentences")
    return create_word_cloud(data,"h", "sentences")



def create_word_cloud(text, cloud_orientation, text_type ):
    if cloud_orientation=="v":
        mask = np.array(Image.open('./imgs/cloud_v.png'))

    elif cloud_orientation=="h":
        mask = np.array(Image.open('./imgs/cloud.png'))

    if text_type=="words":
        wc = WordCloud(background_color='rgba(0,0,0,0)',  mode="RGBA",width=mask.shape[1], height=mask.shape[0], random_state=False, mask=mask) #
    elif text_type=="sentences":
        wc = WordCloud(background_color='rgba(0,0,0,0)',  mode="RGBA", width=mask.shape[1], height=mask.shape[0], random_state=False, mask=mask,max_font_size=50, min_font_size=10) #

    wc.fit_words(text)
    return wc.to_image()

def filter_data(df, band_names_f,genres_f, country_f, album_title_f, song_title_f, song_lyric_f,
                album_year_f, activity_f, song_len_f):

    band_names_f=list(band_names_f)
    genres_f=list(genres_f)
    country_f=list(country_f)
    album_title_f=list(album_title_f)
    song_title_f = list(song_title_f)
    song_lyric_f = list(song_lyric_f)

    album_year_f=list(album_year_f)
    activity_f=list(activity_f)
    song_len_f=[song_len_f]

    #print(band_names_f)
    df_l_f=[]
    if band_names_f!=[]:
        for band_name in band_names_f:
            df_temp=df[df["band name"]==band_name]
            df_l_f.append(df_temp)
    else:
        df_l_f.append(df)


    #print("genre",genres_f)
    if  genres_f!=[]:
        if len(df_l_f)>0:
            for i,df_ in enumerate(df_l_f):
                df_temp_l=[]
                for genre in genres_f:
                    df_temp_l.append(df_[df_["genres"].str.contains(genre)])
            temp_df= pd.concat(df_temp_l)
            df_l_f[i] = temp_df
        else:
            for genre in genres_f:
                df_temp=df[df["genres"].str.contains(genre)]
                df_l_f.append(df_temp)

    #print("country",country_f)
    if  country_f!=[]:
        if len(df_l_f)>0:
            for i,df_ in enumerate(df_l_f):
                df_temp_l=[]
                for country in country_f:
                    df_temp_l.append(df_[df_["country"]==country])
            temp_df= pd.concat(df_temp_l)
            df_l_f[i] = temp_df
        else:
            for country in country_f:
                df_temp=df[df["country"].str.contains(genre)]
                df_l_f.append(df_temp)

    #for sentiment filters
    def sent_to_label(df_temp_, column_, filter_):
        df_temp=pd.DataFrame([], columns=df_temp_.columns)
        if filter_=="neutral":
            df_temp= df_temp_[df_temp_[column_] == 0]
        elif filter_=="positive":
            df_temp= df_temp_[df_temp_[column_] > 0]
        elif filter=="negative":
            df_temp= df_temp_[df_temp_[column_] < 0]
        #print("CHECK",df_temp, filter_)

        return df_temp

    if album_title_f != []:
        if len(df_l_f) > 0:
            for i, df_ in enumerate(df_l_f):
                df_temp_l = []
                for album_title_label in album_title_f:
                    df_temp_l.append(sent_to_label(df_, "album title sentiment", album_title_label))
            temp_df = pd.concat(df_temp_l)
            df_l_f[i] = temp_df
        else:
            for album_title_label in album_title_f:
                df_temp = sent_to_label(df, "album title sentiment", album_title_label)
                df_l_f.append(df_temp)

    if song_title_f != []:
        if len(df_l_f) > 0:
            for i, df_ in enumerate(df_l_f):
                df_temp_l = []
                for song_title_label in song_title_f:
                    df_temp_l.append(sent_to_label(df_, "song title sentiment", song_title_label))
            temp_df = pd.concat(df_temp_l)
            df_l_f[i] = temp_df
        else:
            for song_title_label in song_title_f:
                df_temp = sent_to_label(df, "song title sentiment", song_title_label)
                df_l_f.append(df_temp)

    if song_lyric_f != []:
        if len(df_l_f) > 0:
            for i, df_ in enumerate(df_l_f):
                df_temp_l = []
                for song_lyric_label in song_lyric_f:
                    df_temp_l.append(sent_to_label(df_, "song lyric sentiment", song_lyric_label))
            temp_df = pd.concat(df_temp_l)
            df_l_f[i] = temp_df
        else:
            for song_lyric_label in song_lyric_f:
                df_temp = sent_to_label(df, "song lyric sentiment", song_lyric_label)
                df_l_f.append(df_temp)

    if  album_year_f!=[min(album_years_opts), max(album_years_opts)]:
        if len(df_l_f)>0:
            for i,df_ in enumerate(df_l_f):
                temp_df=df_[(df_["album year"]>=album_year_f[0]) & (df_["album year"]<=album_year_f[1])]
            df_l_f[i] = temp_df
        else:
            df_temp=df[(df["album year"]>=album_year_f[0]) & (df["album year"]<=album_year_f[1])]
            df_l_f.append(df_temp)

    if  activity_f!=[min(activity_opts), max(activity_opts)]:
        print("ACT", activity_f)
        if len(df_l_f)>0:
            temp_df=[]
            for i,df_ in enumerate(df_l_f):
                df_filt=df_.drop_duplicates(subset=["band name"]).reset_index(drop=True)
                #print("FILT DF", df_filt)
                act_list = [[y.split("-") for y in x.split(";")] for x in df_filt["activity"]]
                #print("LIST", act_list)
                act_starts = [[int(z[0]) for z in [y for y in x]] for x in act_list]
                #print("STARTS", act_starts)
                act_ends = [[int(datetime.datetime.now().year) if len(z) == 1 else int(z[1]) for z in [y for y in x]] for x
                            in act_list]
                #print("ENDS", act_ends)

                filt_names = []
                for j, start in enumerate(act_starts):
                    for j2, s in enumerate(start):
                        e = act_ends[j][j2]
                        if s >= activity_f[0] and e <= activity_f[1] and df_filt["band name"][j] not in filt_names:
                            filt_names.append(df_filt["band name"][j])
                #print("NAMES", filt_names)

                temp_df= df_[df_["band name"].str.contains("|".join(filt_names))]
            df_l_f[i] = temp_df
        else:
            df_filt=df.drop_duplicates(subset=["band name"]).reset_index(drop=True)
            #print("FILT DF", df_filt)
            act_list = [[y.split("-") for y in x.split(";")] for x in df_filt["activity"]]
            #print("LIST", act_list)
            act_starts = [[int(z[0]) for z in [y for y in x]] for x in act_list]
            #print("STARTS", act_starts)
            act_ends = [[int(datetime.datetime.now().year) if len(z)==1 else int(z[1]) for z in [y for y in x]] for x in act_list]
            #print("ENDS", act_ends)

            act_ends = [[y[1].strip() for y in x] for x in act_list]

            filt_names = []
            for j, start in enumerate(act_starts):
                for j2, s in enumerate(start):
                    e = act_ends[j][j2]
                    if s >= activity_f[0] and e <= activity_f[1] and df_filt["band name"][j] not in filt_names:
                        filt_names.append(df_filt["band name"][j])
            #print("NAMES", filt_names)

            temp_df = df[df["band name"].str.contains("|".join(filt_names))]
            df_l_f.append(temp_df)

    if song_len_f!=[max(songs_lens)]:
        if len(df_l_f)>0:
            for i,df_ in enumerate(df_l_f):
                temp_df=df_[(df_["song len"]<=song_len_f[0])]
            df_l_f[i] = temp_df
        else:
            temp_df = df[(df["song len"] <= song_len_f[0])]
            df_l_f.append(df_temp)


    if len(df_l_f)>0:
        df_final=pd.concat(df_l_f).reset_index(drop=True)
    else:
        df_final=df
    return df_final
#____________________________________________

#function data


#____
#html data
genres_opts=list(set([y for x in df["genres"].unique() for y in x.split(";")]))
genres_opts=[{"label": x, "value": x} for x in genres_opts]

album_years_opts=  df[df["album year"]>1900]["album year"]
album_yr_minmax= [min(album_years_opts), max(album_years_opts)]

activity_opts= [ [y for y in x.split(";")] for x in df["activity"]]
activity_opts= [y for x in activity_opts for y in x]
activity_opts= [ [y for y in x.split("-")] for x in activity_opts]
activity_opts=[int(y.strip()) for x in activity_opts for y in x]






songs_lens=sorted([x for x in df["song len"]])

songs_lens_markers= dict()

for x,i in enumerate(songs_lens):
    if int(x) in [min(songs_lens),max(songs_lens)] :
       songs_lens_markers[int(x)]= str(round(x/60,2)).replace(".",":")
    #elif i==len(songs_lens)//2: #not working
    #    songs_lens_markers[int(x)] = str(round(x/60),2).replace(".",":")

    else:
        songs_lens_markers[int(x)]= ""

#_________________________________________________________________________________________________
#LAYOUT

app.layout = html.Div(style={"backgroundColor":"rgba(17,17,17,1)", "color":"white", "fontFamily":"Arial"},children=[
dcc.ConfirmDialog(
        id='confirm-dialog',
        message='''\u26A0 IMPORTANT \u26A0\n
         \u26A0 Any lag is due to the hosting service free tier (i.e. render.com).\n
         \u26A0 ON START, first choose at least one filter, then press "Apply".\n
         \u26A0 Clicking on other buttons first will make the app crash.\n
         \u26A0 Applying without any filter is not recommended.\n
         \u26A0 Press "Generate clouds" with extra care: slow function.\n
         \u26A0 Save clouds with right click.\n
         \u26A0 Save other plots through their upper bar.''' ,
    ),
    html.H1("Music dashboard", style={"textAlign" : "center"}),
    html.Div(id="filters_metacont",
             style={"top":"0", "position":"sticky", "zIndex":"99999",
                    "backgroundColor": "rgba(255,255,255,0.8)", "color":"black", "textAlign":"center",
                   }, children=[

    html.Button("Hide filters", id="filters_btn", n_clicks=0,
                style={"width":"100%", "backgroundColor":"rgba(0,122,204,1)","color":"white",
                       "padding":"0.3%"}),
        html.Div(id="filters_cont", style={}, children=[ #style={"height":"0%", "display":"none"}
            html.Div(id="slctr_subcont1_labels",  style={"display":"flex", "flexDirection":"row"} ,children=[
                    html.Label("Band name",className="label",style={"width":"100%"}),
                    html.Label("Genre", className="label", style={"width": "100%"}),
                    html.Label("Country", className="label", style={"width": "100%"}),
                    html.Label("Album title", className="label", style={"width": "100%"}),
                    html.Label("Song title", className="label", style={"width": "100%"}),
                    html.Label("Song lyric", className="label", style={"width": "100%"}),

            ]),

                    html.Div(id="slctr_subcont1_inps",
                             style={"display":"flex", "flexDirection":"row","border":"1px solid black"} ,children=[

                        dcc.Dropdown(id="band_name_slctr", options=[{"label": x, "value": x} for x in df["band name"].unique()],
                                     multi=True, value="",
                                     style={"width": "100%"})
                         ,

                        dcc.Dropdown(id="genre_slctr", options=genres_opts, multi=True,
                                     value="",
                                     style={"width": "100%"}),

                        dcc.Dropdown(id="country_slctr", options=[{"label": x, "value": x} for x in df["country"].unique()], multi=True,
                                     value="",
                                     style={"width": "100%"}),

                        dcc.Dropdown(id="album_title_slctr", options=[{"label": x, "value": x} for x in ["neutral", "positive", "negative"]],
                                     multi=True,
                                     value="",
                                     style={"width": "100%"}),

                        dcc.Dropdown(id="song_title_slctr", options=[{"label": x, "value": x} for x in ["neutral", "positive", "negative"]],
                                     multi=True,
                                     value="",
                                     style={"width": "100%"}),
                        dcc.Dropdown(id="song_lyric_slctr",
                                     options=[{"label": x, "value": x} for x in ["neutral", "positive", "negative"]],
                                     multi=True,
                                     value="",
                                     style={"width": "100%"}),
                    ]),





                html.Label("Album year", className="label", style={"width": "25%"}),

                dcc.RangeSlider(album_yr_minmax[0],album_yr_minmax[1], id="album_year_slctr",
                                value=[album_yr_minmax[0],album_yr_minmax[1]],
                                marks={int(x):str(x) for x in album_years_opts },
                                step=5,
                                tooltip={"placement": "top", "always_visible": False}

                                ),

                html.Label("Activity interval", className="label", style={"width": "25%"}),
                dcc.RangeSlider(min(activity_opts), max(activity_opts), id="activity_slctr",
                                value=[min(activity_opts),max(activity_opts)],
                                marks={int(x): str(x) for x in activity_opts},
                                step=5,
                                tooltip={"placement": "top", "always_visible": False}

                                ),

                html.Label("Song length", className="label", style={"width": "25%"}),
                dcc.Slider(min(songs_lens), max(songs_lens), id="song_len_slctr",
                                value=max(songs_lens),
                                marks=songs_lens_markers,
                                tooltip={"placement": "bottom", "always_visible": False}

                                ),
                html.Button("Apply",id="start_btn", n_clicks=0,
                            style={"visibility":"visible", "backgroundColor":"rgba(255,0,0,0.7)", "color":"white", "fontSize":"20px"} ),
                dcc.Loading(
                id="loading-input_start", style={"zIndex":"11000"},
                children=[html.Span([html.Span(id="loading-output_start",
                                               style={"position": "relative", "width": "50%", "height": "50%",
                                                      "backgroundColor": "transparent", "color": "transparent",


                                                      }

                                               )])],
                type="default",
            ),

        ]),

    ])
    ,

    html.Button("Generate sentiment clouds", id="generate_clouds_btn",
                style={"width":"85%", "marginLeft":"5%",
                       "backgroundColor":"rgba(0,122,204,1)", "color":"white",
                       "textAlign":"center","padding":"0.3%",
                       "marginBottom":"1vh"}, n_clicks=0 ),

    dcc.Loading(
        id="loading-input_gencloud", style={"zIndex":"11000"},
        children=[html.Span([html.Span(id="loading-output_gencloud",
                                       style={"position": "relative", "width": "50%", "height": "50%",
                                              "backgroundColor": "transparent", "color": "transparent",
                                              "zIndex": "100000",

                                              }

                                       )])],
        type="default",
    ),
    html.Div(id="graphs_metacont", style={"display":"flex", "flexDirection": "row", "width":"100%","height":"100%", "backgroundColor":"rgb(17, 17, 17)"}, children=[

        html.Div(id="general_graphs_cont", className="subcont", style={"width":"100%", "height":"auto",
                                                                       "display":"flex", "flexDirection":"column","alignContent":"flex-end"}, children=[

           html.Div(id="upper_general_cont",style={"width":"100%","height":"100%"}, children=[
                html.Button("Show single counters", id="singlecounters_btn",
                style={"width":"75%", "marginLeft":"10%",
                       "backgroundColor":"rgba(0,122,204,1)", "color":"white",
                       "textAlign":"center","padding":"0.3%"}, n_clicks=0 ),
                dcc.Loading(
                id="loading-input_donuts", style={"zIndex":"11000"},
                children=[html.Span([html.Span(id="loading-output_donuts",
                                               style={"position": "relative", "width": "50%", "height": "50%",
                                                      "backgroundColor": "transparent", "color": "transparent",


                                                      }

                                               )])],
                type="default",
            ),
               html.Div(id="donuts_cont", style={"width": "100%", "height":"50%", "display":"none" }, children=[
                   html.Div(id="donuts_subcont1",style={"width": "100%", "height":"50%", "display":"flex", "flexDirection":"row"},
                            children=[
                        dcc.Graph(id="bandname_donut", figure={}, style={"width":"20%","height":"50vh"}),
                       dcc.Graph(id="country_donut", figure={}, style={"width":"20%","height":"50vh"}),
                       dcc.Graph(id="genres_donut", figure={}, style={"width":"20%","height":"50vh"}),
                       dcc.Graph(id="albumtitle_donut", figure={}, style={"width":"20%","height":"50vh"}),
                       dcc.Graph(id="songtitle_donut", figure={}, style={"width": "20%","height":"50vh"}),
                   ]),

                   html.Div(id="donuts_subcont2", style={"width": "100%", "height":"25%", "display":"flex", "flexDirection":"row"},
                            children=[

                       dcc.Graph(id="albumyear_hist", figure={}, style={"width":"20%","height":"25vh"}),
                   dcc.Graph(id="songlen_hist", figure={}, style={"width": "20%","height":"25vh"}),
                   dcc.Graph(id="albumtitlesent_hist", figure={}, style={"width": "20%","height":"25vh"}),
                   dcc.Graph(id="songtitlesent_hist", figure={}, style={"width": "20%","height":"25vh"}),
                   dcc.Graph(id="lyricsent_hist", figure={}, style={"width": "20%","height":"25vh"}),
                ]),

               ]),

               html.Div(id="map_cont", style={"width":"100%","height":"100%"},children=[
                    dcc.Graph(id="chor_g",figure=px.histogram(x=[],template="plotly_dark"))
               ]),
           ]),

            html.Button("Show marginal distributions", id="marginal_btn", n_clicks=0,
                        style={"width": "75%", "marginLeft": "10%", "backgroundColor":"rgba(0,122,204,1)", "color":"white",
                               "textAlign":"center", "padding":"0.3%"}
                        ),
            dcc.Loading(
                id="loading-input_marginal", style={"zIndex":"11000"},
                children=[html.Span([html.Span(id="loading-output_marginal",
                                               style={"position": "relative", "width": "50%", "height": "50%",
                                                      "backgroundColor": "transparent", "color": "transparent",


                                                      }

                                               )])],
                type="default",
            ),

            html.Div(id="lower_general_cont", style={"display":"flex", "flexDirection":"row","width":"100%","height":"100%","float":"bottom", "alignContent":"flex-end"}, children=[

                html.Div(id="gnatt_cont", style={"width":"50%"}, children=[
                    html.Div(id="marginal_x_cont", hidden=True, children=[
                        dcc.Graph(id="gnatt_marginal_x",  figure={}),

                    ]),
                    dcc.Graph(id="gnatt_g", figure=px.histogram(x=[],template="plotly_dark")),
                ]),
                dcc.Loading(
                    id="loading-input_metacont", style={"zIndex":"11000"},
                    children=[html.Div(style={"position":"absolute"}, children=[html.Div(id="loading-output_metacont",
                                                 style={"position": "relative",
                                                        "backgroundColor": "transparent", "color": "transparent",


                                                        }

                                                 )])],
                    type="default",
                ),
                html.Div(id="heats_cont", style={"width": "50%", "float":"bottom"}, children=[
                    dcc.Graph(id="heats_g", figure=px.histogram(x=[],template="plotly_dark")),
                    html.Div(id="marginal_y_cont", hidden=True, children=[
                        dcc.Graph(id="gnatt_marginal_y", figure={}),
                    ]),
                ]),

            ]),

        ]),


        html.Div(id="clouds_cont", className="subcont", hidden=True, style={ "width":"0%","height":"0%",
                                                               "padding":"2%"}, children=[
            html.Label("Words", id="word label", style={"color": "white", "marginLeft": "45%"}),
            html.Div(id="w_clouds_cont", className="subcont", style={"display": "flex", "flexDirection":"row", "height":"50%", "width":"100%"}, children=[
                html.Div(id="w_posneg_clouds_cont", className="subcont", style={"display": "flex", "flexDirection": "column","width": "50%", "height": "50%","padding": "5%"},
                         children=[
                             html.Label("Positive", id="positive word label", style={"color": "white", "marginLeft": "45%"}),
                             html.Img(id="w_positive_cloud", style={"width":"80%"} , src=""),
                             html.Label("Negative", id="negative word label",style={"color": "white", "marginLeft": "45%"}),
                             html.Img(id="w_negative_cloud", style={"width":"80%"} , src="")
                         ]),
                html.Div(id="w_neutral_clouds_cont", className="subcont",
                         style={"width": "50%", "height": "100%", "padding": "5%", "dispaly":"flex", "flexDirection":"column"},
                         children=[
                             html.Label("Neutral", id="neutral word label",
                                        style={"color": "white","marginLeft": "25%"}),
                             html.Img(id="w_neutral_cloud", style={"width": "80%"}, src="")
                         ]),
            ]),

            html.Label("Sentences", id="sent label", style={"color": "white", "marginLeft": "45%"}),
            html.Div(id="s_clouds_cont", className="subcont",
                     style={"display": "flex", "flexDirection": "row", "height": "50%", "width": "100%",  "float":"bottom"}, children=[
                    html.Div(id="s_posneg_clouds_cont", className="subcont",
                             style={"display": "flex", "flexDirection": "column","width": "50%", "height": "50%","padding": "5%" },
                             children=[
                                 html.Label("Positive", id="positive sent label",style={"color": "white", "marginLeft": "45%"}),
                                 html.Img(id="s_positive_cloud", style={"width": "80%"}, src=""),
                                 html.Label("Negative", id="negative sent label",style={"color": "white", "marginLeft": "45%"}),
                                 html.Img(id="s_negative_cloud", style={ "width": "80%"}, src="")
                             ]),
                    html.Div(id="s_neutral_clouds_cont", className="subcont",
                             style={"width": "50%", "height": "100%", "dispaly":"flex", "flexDirection":"column","padding": "5%"},
                             children=[
                                 html.Label("Neutral", id="neutral sent label",
                                            style={"color": "white","marginLeft": "25%"}),
                                 html.Img(id="s_neutral_cloud", style={"width": "80%"}, src="")
                             ]),
                ]),

        ])



    ]),

])

#_______________________________________________________________________________________________________________
#CALLBACKS

"""
#CLOUDS
 Output(component_id="w_neutral_cloud", component_property="src"),
  Output(component_id="w_positive_cloud", component_property="src"),
  Output(component_id="w_negative_cloud", component_property="src"),
  Output(component_id="s_neutral_cloud", component_property="src"),
  Output(component_id="s_positive_cloud", component_property="src"),
  Output(component_id="s_negative_cloud", component_property="src"),

  Output(component_id="general_graphs_cont", component_property="style",allow_duplicate=True),
  Output(component_id="clouds_cont", component_property="style", allow_duplicate=True),
  Output(component_id="clouds_cont", component_property="hidden", allow_duplicate=True),
  Output(component_id="generate_clouds_btn", component_property="disabled"),
 Output(component_id="loading-output_gencloud", component_property="children")
 """

"""
#CLOUDS
#Input(component_id="generate_clouds_btn", component_property="n_clicks"),
"""




@app.callback([

               #start_btn
               Output(component_id="confirm-dialog", component_property="displayed"),
               #Output(component_id="loading-output_start", component_property="children")
               #Output(component_id="", component_property="")




],
              [#filters
               Input(component_id="band_name_slctr", component_property="value"),
               Input(component_id="genre_slctr", component_property="value"),
               Input(component_id="country_slctr", component_property="value"),
               Input(component_id="album_title_slctr", component_property="value"),
               Input(component_id="song_title_slctr", component_property="value"),
               Input(component_id="song_lyric_slctr", component_property="value"),
               Input(component_id="album_year_slctr", component_property="value"),
               Input(component_id="activity_slctr", component_property="value"),
               Input(component_id="song_len_slctr", component_property="value"),
               Input(component_id="confirm-dialog", component_property="submit_n_clicks"),

                  #Input(component_id="loading-output_start", component_property="children"),

               #MAP
               #GNATT

               #HEATS

               ],
              ) #prevent_initial_call=True
def update_filters(band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f, album_year_f,activity_f, song_len_f,
                   n_click_start): #  n_click_clouds

    global filtered_df
    filtered_df = filter_data(df, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                            album_year_f,activity_f, song_len_f)
    if type(n_click_start)!=int:
        n_click_start=0

    if n_click_start==0:
        return [True]
    elif n_click_start>0:
        return [False]


@app.callback(
            [
             #MAP
               Output(component_id="chor_g", component_property="figure"),
               Output(component_id="general_graphs_cont", component_property="style"),
               Output(component_id="clouds_cont", component_property="style"),
               Output(component_id="clouds_cont", component_property="hidden"),
               #GNATT
               Output(component_id="loading-output_metacont", component_property="children"),
               Output(component_id="gnatt_g", component_property="figure"),
               #HEATS
               Output(component_id="heats_g", component_property="figure"),
                #MARGINAL
                Output(component_id="marginal_x_cont", component_property="hidden", allow_duplicate=True),
                Output(component_id="marginal_y_cont", component_property="hidden", allow_duplicate=True),
                Output(component_id="gnatt_marginal_x", component_property="figure", allow_duplicate=True),
                Output(component_id="gnatt_marginal_y", component_property="figure", allow_duplicate=True),
                Output(component_id="marginal_btn", component_property="children", allow_duplicate=True),
                Output(component_id="marginal_btn", component_property="disabled", allow_duplicate=True),
                Output(component_id="loading-output_marginal", component_property="children", allow_duplicate=True),

                Output(component_id="bandname_donut", component_property="figure"),
                Output(component_id="country_donut", component_property="figure"),
                Output(component_id="genres_donut", component_property="figure"),
                Output(component_id="albumtitle_donut", component_property="figure"),
                Output(component_id="songtitle_donut", component_property="figure"),
                Output(component_id="albumyear_hist", component_property="figure"),
                Output(component_id="songlen_hist", component_property="figure"),
                Output(component_id="albumtitlesent_hist", component_property="figure"),
                Output(component_id="songtitlesent_hist", component_property="figure"),
                Output(component_id="lyricsent_hist", component_property="figure"),

                Output(component_id="donuts_cont", component_property="style"),
                Output(component_id="singlecounters_btn", component_property="children", allow_duplicate=True),
                Output(component_id="loading-output_donuts", component_property="children", allow_duplicate=True),


                Output(component_id="start_btn", component_property="style", allow_duplicate=True),
               Output(component_id="loading-output_start", component_property="children", allow_duplicate=True)

    ],
    [
        Input(component_id="graphs_metacont", component_property="id"),
        # start button
        Input(component_id="start_btn", component_property="n_clicks"),
        Input(component_id="start_btn", component_property="id"),
        Input(component_id="gnatt_cont", component_property="id"),
        Input(component_id="loading-input_donuts", component_property="children")
    ],
    prevent_initial_call=True
)
def draw_general_plots(metacont_loading_val,n_clicks_start, loading_start, loading_val_gnatt, loading_val_donuts):
    global filtered_df

    singleplots_data= [go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers"))]*10
    singleplots_data.append({"width": "100%", "height":"50%", "display":"none"})
    singleplots_data.append("Show single counters")
    singleplots_data.append(loading_val_donuts)
    map_data = update_map(filtered_df)
    gnatt_data = update_gnatt(filtered_df, loading_val_gnatt)
    heats_data = update_heats(filtered_df)
    marginals_data = [True, True,
                      go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")),
                      go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")),
                      "Show marginal distributions", False, ""]
    # marginals_data= update_marginals( n_click_marginals, loading_val_marginals)
    start_btn_style = {"visibility":"visible", "backgroundColor":"rgba(255,0,0,0.7)", "color":"white", "fontSize":"20px"}
    final_data = [
                    *map_data, *gnatt_data, *heats_data, *marginals_data, *singleplots_data,
                   start_btn_style, loading_start,
                 ]
    #(final_data)
    return final_data



@app.callback(
    [Output(component_id="filters_cont", component_property="hidden"),
     Output(component_id="filters_cont", component_property="style"),
     Output(component_id="filters_btn", component_property="children")
     ],
    [Input(component_id="filters_btn", component_property="n_clicks")],
    prevent_initial_call=True
)
def update_filters_visibility(n_clicks):
    if n_clicks%2!=0:
        return [True,{"height":"100%","transition":"2s"}, "Show filters"]
    else:
        return [False,{"height":"0%","transition":"2s"}, "Hide filters"]

def update_map(filtered_df):
    #document.getElementById()
    return [create_map(filtered_df),{"width":"100%", "height":"100%"},{"width":"0%", "height":"auto"},True]

def update_gnatt(filtered_df, loading_val):
    return [loading_val, create_gnatt(filtered_df)]


def update_heats(filtered_df):
    return [create_heats(filtered_df)]


@app.callback(
    [ Output(component_id="bandname_donut", component_property="figure", allow_duplicate=True),
                Output(component_id="country_donut", component_property="figure", allow_duplicate=True),
                Output(component_id="genres_donut", component_property="figure", allow_duplicate=True),
                Output(component_id="albumtitle_donut", component_property="figure", allow_duplicate=True),
                Output(component_id="songtitle_donut", component_property="figure", allow_duplicate=True),
                Output(component_id="albumyear_hist", component_property="figure", allow_duplicate=True),
                Output(component_id="songlen_hist", component_property="figure", allow_duplicate=True),
                Output(component_id="albumtitlesent_hist", component_property="figure", allow_duplicate=True),
                Output(component_id="songtitlesent_hist", component_property="figure", allow_duplicate=True),
                Output(component_id="lyricsent_hist", component_property="figure", allow_duplicate=True),

                Output(component_id="donuts_cont", component_property="style", allow_duplicate=True),
                Output(component_id="singlecounters_btn", component_property="children", allow_duplicate=True),
                Output(component_id="loading-output_donuts", component_property="children", allow_duplicate=True)

    ],

    [
        Input(component_id="singlecounters_btn", component_property="n_clicks"),
        Input(component_id="loading-input_donuts", component_property="children")

    ],
    prevent_initial_call=True
)
def update_single_counters(n_clicks, loading_val):
    global filtered_df
    singleplots_data= create_single_stats(filtered_df)
    if n_clicks%2==0:
        style_={"width": "100%", "height":"50%", "display":"none" }
        text_="Show single counters"
    elif n_clicks%2!=0:
        style_={"width": "100%", "height":"50%", "display":"flex", "flexDirection":"column" }
        text_="Hide single counters"

    return [*singleplots_data, style_, text_, loading_val]

@app.callback(
    [
#MARGINAL
               Output(component_id="marginal_x_cont", component_property="hidden",allow_duplicate=True),
               Output(component_id="marginal_y_cont", component_property="hidden",allow_duplicate=True),
               Output(component_id="gnatt_marginal_x", component_property="figure",allow_duplicate=True),
               Output(component_id="gnatt_marginal_y", component_property="figure",allow_duplicate=True),
               Output(component_id="marginal_btn", component_property="children",allow_duplicate=True),
               Output(component_id="marginal_btn", component_property="disabled",allow_duplicate=True),
               Output(component_id="loading-output_marginal", component_property="children",allow_duplicate=True),
    ],
    [
        # MARGINALS
        Input(component_id="marginal_btn", component_property="n_clicks"),
        Input(component_id="marginal_btn", component_property="id"),
    ],
    prevent_initial_call=True
)
def update_marginals(n_clicks, loading_val):
    global filtered_df
    if list(dash.callback_context.triggered_prop_ids.values())==["marginal_btn"]:
        if n_clicks%2!=0:
            return[False, False, create_gnatt_marginal_x(filtered_df), create_gnatt_marginal_y(filtered_df), "Hide marginal distributions", False, loading_val]
        else:
            #filtered_df = filter_data(df[0:2], band_names_f, genre_slctr_f, country_f, album_title_f) #only one row, faster load and hide marginals
            return [True, True,
                go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")),
                go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")), "Show marginal distributions", False, loading_val] #
    else:
        #filtered_df = filter_data(df[0:2], band_names_f, genre_slctr_f, country_f, album_title_f)  # only one row, faster load and hide marginals
        return [True, True,
                go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")),
                go.Figure(go.Scatter(x=pd.Series(dtype=object), y=pd.Series(dtype=object), mode="markers")), "Show marginal distributions" , False, loading_val]  #


@app.callback(
    [Output(component_id="w_neutral_cloud", component_property="src"),
     Output(component_id="w_positive_cloud", component_property="src"),
     Output(component_id="w_negative_cloud", component_property="src"),
     Output(component_id="s_neutral_cloud", component_property="src"),
     Output(component_id="s_positive_cloud", component_property="src"),
     Output(component_id="s_negative_cloud", component_property="src"),
     Output(component_id="general_graphs_cont", component_property="style",allow_duplicate=True),
     Output(component_id="clouds_cont", component_property="style", allow_duplicate=True),
     Output(component_id="clouds_cont", component_property="hidden", allow_duplicate=True),
     Output(component_id="generate_clouds_btn", component_property="disabled"),
     Output(component_id="loading-output_gencloud", component_property="children")
     ],
    [Input(component_id="generate_clouds_btn", component_property="n_clicks"),
     Input(component_id="generate_clouds_btn", component_property="id"),
     Input(component_id="band_name_slctr", component_property="value"),
    Input(component_id="genre_slctr", component_property="value"),
    Input(component_id="country_slctr", component_property="value"),
    Input(component_id="album_title_slctr", component_property="value"),
     Input(component_id="song_title_slctr", component_property="value"),
     Input(component_id="song_lyric_slctr", component_property="value"),
    Input(component_id="album_year_slctr", component_property="value"),
    Input(component_id="activity_slctr", component_property="value"),
Input(component_id="song_len_slctr", component_property="value")
     ],
    prevent_initial_call=True,
)
def generate_clouds(n_clicks, loading_val, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                    album_year_f, activity_f, song_len_f):
    global filtered_df
    if list(dash.callback_context.triggered_prop_ids.values())!=["generate_clouds_btn"] or filtered_df.empty:
        print("CLOUD PREVENTED")
        return [Image.new("RGB", (800, 1280), (255, 255, 255)),
                Image.new("RGB", (800, 1280), (255, 255, 255)),
                Image.new("RGB", (800, 1280), (255, 255, 255)),
                Image.new("RGB",  (800, 1280), (255, 255, 255)),
                Image.new("RGB",  (800, 1280), (255, 255, 255)),
                Image.new("RGB",  (800, 1280), (255, 255, 255)),

            {"width": "100%", "height": "100%"},
            {"width": "0%", "height": "auto"},
            False, #should be True
            False,
            loading_val
           ]
    print("clicked")
    filtered_data=process_text(filtered_df,"all")
    print("finished process")
    img1=create_neutral_word_cloud(filtered_df,filtered_data)
    #img1.save("./temp imgs/neutral words cloud.png",dpi=(300, 300))
    img2=create_positive_word_cloud(filtered_df,filtered_data)
    #img2.save("./temp imgs/positive words cloud.png",dpi=(300, 300))
    img3=create_negative_word_cloud(filtered_df,filtered_data)
    #img3.save("./temp imgs/negative words cloud.png",dpi=(300, 300))

    img4=create_neutral_sent_cloud(filtered_df,filtered_data)
    #img4.save("./temp imgs/neutral sents cloud.png",dpi=(300, 300))
    img5=create_positive_sent_cloud(filtered_df,filtered_data)
    #img5.save("./temp imgs/positive sents cloud.png",dpi=(300, 300))
    img6=create_negative_sent_cloud(filtered_df,filtered_data)
    #img6.save("./temp imgs/negative sents cloud.png",dpi=(300, 300))



    return[img1,#Image.open("./temp imgs/neutral words cloud.png"),
           img2,#Image.open("./temp imgs/positive words cloud.png"),
           img3,#Image.open("./temp imgs/negative words cloud.png"),
           img4,#Image.open("./temp imgs/neutral sents cloud.png"),
           img5,#Image.open("./temp imgs/positive sents cloud.png"),
           img6,#Image.open("./temp imgs/negative sents cloud.png"),

           {"width":"60%", "height":"100%"},
           {"width":"40%", "height":"auto"},
           False,
           False, #True if you want to block button AFTER clouds on screen
           loading_val
           ]





if __name__=="__main__":
    app.run_server(debug=True)