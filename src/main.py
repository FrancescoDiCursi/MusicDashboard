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
from nltk.corpus import stopwords
import re

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import flask
from tqdm import tqdm



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

filtered_data=[]



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
    print("MARGINAL DF:  ", df)
    df_scatter_x = create_scatter_dfg_activity(df)

    print("marginal x", df_scatter_x)
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


def process_text(df,type_):
    positive_words, negative_words, neutral_words , positive_sentences, negative_sentences, neutral_sentences = [],[],[],[],[],[]

    stopwords_ = [x for x in stopwords.words("english")]
    df_sentiment = df.copy().drop_duplicates(subset=["song lyric"], keep="first")
    df_sentiment_lyrics = df_sentiment["song lyric"]
    df_sentiment_lyrics_sents = [[y.strip() for y in str(x).split("\n")] for x in df_sentiment_lyrics if x != ""]
    df_sentiment_lyrics_sents = [[y.lower() for y in x if y != ""] for x in df_sentiment_lyrics_sents]

    df_sentiment_lyrics_words = [[re.sub("\W", " ", y).split(" ") for y in x] for x in df_sentiment_lyrics_sents]
    df_sentiment_lyrics_words = [
        [[z.lower() for z in y if z != "" and z.lower() not in stopwords_] for y in x if y != []] \
        for x in df_sentiment_lyrics_words]
    # freqs
    word_dict = {z: [0] for x in df_sentiment_lyrics_words for y in x for z in y}
    print("calculating words sentiment")
    for x in tqdm(df_sentiment_lyrics_words):
        for y in x:
            for z in y:
                word_dict[z][0] += 1
                word_dict[z] = [word_dict[z][0], TextBlob(z).sentiment.polarity]
    sents_dict = {y: [0] for x in df_sentiment_lyrics_sents for y in x}
    print("calculating sentences sentiment")
    for x in tqdm(df_sentiment_lyrics_sents):
        for y in x:
            splitted_sent = [x for x in re.sub("\W", " ", y).split(" ") if x != ""]
            sentence_score = 0
            for z in splitted_sent:
                if z not in stopwords_:
                    sentence_score += word_dict[z][0]
            sents_dict[y] = [sentence_score, TextBlob(y).sentiment.polarity]

    sorted_word_dict_keys=sorted(word_dict.items(), key=lambda l: l[1][0], reverse=True)
    sorted_sents_dict_keys=sorted(sents_dict.items(), key=lambda l: l[1][0], reverse=True)
            #
    positive_words = {x[0]: x[1][0] for x in sorted_word_dict_keys if
                      x[1][1] > 0}
    negative_words = {x[0]: x[1][0] for x in sorted_word_dict_keys if
                      x[1][1] < 0}
    neutral_words = {x[0]: x[1][0] for x in sorted_word_dict_keys if
                     x[1][1] == 0}
    positive_sentences = {x[0]: x[1][0] for x in  sorted_sents_dict_keys
                          if x[1][1] > 0}
    negative_sentences = {x[0]: x[1][0] for x in  sorted_sents_dict_keys
                          if x[1][1] < 0}
    neutral_sentences = {x[0]: x[1][0] for x in  sorted_sents_dict_keys
                         if x[1][1] == 0}

    return [[positive_words,negative_words, neutral_words], [positive_sentences, negative_sentences, neutral_sentences]]

def create_neutral_word_cloud(df,filtered_data):
    data=filtered_data[0][2]
    print("neutral words")
    return create_word_cloud(data,"v","words")

def create_positive_word_cloud(df,filtered_data):
    data=filtered_data[0][0]
    print("positive words")
    return create_word_cloud(data,"h","words")

def create_negative_word_cloud(df,filtered_data):
    data=filtered_data[0][1]
    print("negative sentences")
    return create_word_cloud(data,"h","words")

#
def create_neutral_sent_cloud(df,filtered_data):
    data= filtered_data[1][2]
    print("neutral sentences")
    return create_word_cloud(data,"v", "sentences")

def create_positive_sent_cloud(df,filtered_data):
    data= filtered_data[1][0]
    print("positive sentences")
    return create_word_cloud(data,"h", "sentences")

def create_negative_sent_cloud(df,filtered_data):
    data= filtered_data[1][1]
    print("negative sentences")
    return create_word_cloud(data,"h", "sentences")



def create_word_cloud(text, cloud_orientation, text_type ):
    if cloud_orientation=="v":
        mask = np.array(Image.open('./imgs/cloud_v.png'))
    elif cloud_orientation=="h":
        mask = np.array(Image.open('./imgs/cloud.png'))
    if text_type=="words":
        wc = WordCloud(background_color='rgba(0,0,0,0)',  mode="RGBA", width=mask.shape[1], height=mask.shape[0], random_state=False, mask=mask)
    elif text_type=="sentences":
        wc = WordCloud(background_color='rgba(0,0,0,0)',  mode="RGBA",  width=mask.shape[1], height=mask.shape[0], random_state=False, mask=mask,max_font_size=50, min_font_size=10)

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

    print(band_names_f)
    df_l_f=[]
    if band_names_f!=[]:
        for band_name in band_names_f:
            df_temp=df[df["band name"]==band_name]
            df_l_f.append(df_temp)
    else:
        df_l_f.append(df)


    print("genre",genres_f)
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

    print("country",country_f)
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
        print("CHECK",df_temp, filter_)

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
                print("FILT DF", df_filt)
                act_list = [[y.split("-") for y in x.split(";")] for x in df_filt["activity"]]
                print("LIST", act_list)
                act_starts = [[int(z[0]) for z in [y for y in x]] for x in act_list]
                print("STARTS", act_starts)
                act_ends = [[int(datetime.datetime.now().year) if len(z) == 1 else int(z[1]) for z in [y for y in x]] for x
                            in act_list]
                print("ENDS", act_ends)

                filt_names = []
                for j, start in enumerate(act_starts):
                    for j2, s in enumerate(start):
                        e = act_ends[j][j2]
                        if s >= activity_f[0] and e <= activity_f[1] and df_filt["band name"][j] not in filt_names:
                            filt_names.append(df_filt["band name"][j])
                print("NAMES", filt_names)

                temp_df= df_[df_["band name"].str.contains("|".join(filt_names))]
            df_l_f[i] = temp_df
        else:
            df_filt=df.drop_duplicates(subset=["band name"]).reset_index(drop=True)
            print("FILT DF", df_filt)
            act_list = [[y.split("-") for y in x.split(";")] for x in df_filt["activity"]]
            print("LIST", act_list)
            act_starts = [[int(z[0]) for z in [y for y in x]] for x in act_list]
            print("STARTS", act_starts)
            act_ends = [[int(datetime.datetime.now().year) if len(z)==1 else int(z[1]) for z in [y for y in x]] for x in act_list]
            print("ENDS", act_ends)

            act_ends = [[y[1].strip() for y in x] for x in act_list]

            filt_names = []
            for j, start in enumerate(act_starts):
                for j2, s in enumerate(start):
                    e = act_ends[j][j2]
                    if s >= activity_f[0] and e <= activity_f[1] and df_filt["band name"][j] not in filt_names:
                        filt_names.append(df_filt["band name"][j])
            print("NAMES", filt_names)

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

app.layout = html.Div(style={"background-color":"rgba(17,17,17,1)", "color":"white", "font-family":"Arial"},children=[
    html.H1("Music dashboard", style={"textAlign" : "center"}),
    html.Div(id="filters_metacont",
             style={"top":"0", "position":"sticky", "z-index":"99999",
                    "background-color": "rgba(255,255,255,0.8)", "color":"black", "text-align":"center",
                   }, children=[

    html.Button("Hide filters", id="filters_btn", n_clicks=0,
                style={"width":"100%", "background-color":"rgba(0,122,204,1)","color":"white",
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

        ]),


    ]),

    html.Button("Generate sentiment clouds", id="generate_clouds_btn",
                style={"width":"75%", "margin-left":"10%",
                       "background-color":"rgba(0,122,204,1)", "color":"white",
                       "text-align":"center","padding":"0.3%"}, n_clicks=0 ),

    dcc.Loading(
        id="loading-input_gencloud", style={"z-index":"11000"},
        children=[html.Span([html.Span(id="loading-output_gencloud",
                                       style={"position": "relative", "width": "50%", "height": "50%",
                                              "background-color": "transparent", "color": "transparent",
                                              "z-index": "100000",

                                              }

                                       )])],
        type="default",
    ),
    html.Div(id="graphs_metacont", style={"display":"flex", "flexDirection": "row", "width":"100%","height":"100%", "background-color":"rgb(17, 17, 17)"}, children=[

        html.Div(id="general_graphs_cont", className="subcont", style={"width":"100%", "height":"auto",
                                                                       "display":"flex", "flex-direction":"column","align-content":"flex-end"}, children=[

           html.Div(id="upper_general_cont",style={"width":"100%","height":"100%"}, children=[
               html.Div(id="map_cont", style={"width":"100%","height":"100%"},children=[
                    dcc.Graph(id="chor_g",figure={})
               ]),
           ]),

            html.Button("Show marginal distributions", id="marginal_btn", n_clicks=0,
                        style={"width": "75%", "margin-left": "10%", "background-color":"rgba(0,122,204,1)", "color":"white",
                               "text-align":"center", "padding":"0.3%"}
                        ),
            dcc.Loading(
                id="loading-input_marginal", style={"z-index":"11000"},
                children=[html.Span([html.Span(id="loading-output_marginal",
                                               style={"position": "relative", "width": "50%", "height": "50%",
                                                      "background-color": "transparent", "color": "transparent",


                                                      }

                                               )])],
                type="default",
            ),

            html.Div(id="lower_general_cont", style={"display":"flex", "flexDirection":"row","width":"100%","height":"100%","float":"bottom", "align-content":"flex-end"}, children=[

                html.Div(id="gnatt_cont", style={"width":"50%"}, children=[
                    html.Div(id="marginal_x_cont", hidden=True, children=[
                        dcc.Graph(id="gnatt_marginal_x",  figure={}),

                    ]),
                    dcc.Graph(id="gnatt_g", figure={}),
                ]),
                dcc.Loading(
                    id="loading-input_metacont", style={"z-index":"11000"},
                    children=[html.Div(style={"position":"absolute"}, children=[html.Div(id="loading-output_metacont",
                                                 style={"position": "relative",
                                                        "background-color": "transparent", "color": "transparent",


                                                        }

                                                 )])],
                    type="default",
                ),
                html.Div(id="heats_cont", style={"width": "50%", "float":"bottom"}, children=[
                    dcc.Graph(id="heats_g", figure={}),
                    html.Div(id="marginal_y_cont", hidden=True, children=[
                        dcc.Graph(id="gnatt_marginal_y", figure={}),
                    ]),
                ]),

            ]),

        ]),


        html.Div(id="clouds_cont", className="subcont", hidden=True, style={ "width":"0%","height":"0%",
                                                               "padding":"2%"}, children=[
            html.Label("Words", id="word label", style={"color": "white", "margin-left": "45%"}),
            html.Div(id="w_clouds_cont", className="subcont", style={"display": "flex", "flex-direction":"row", "height":"50%", "width":"100%"}, children=[
                html.Div(id="w_posneg_clouds_cont", className="subcont", style={"display": "flex", "flex-direction": "column","width": "50%", "height": "50%","padding": "5%"},
                         children=[
                             html.Label("Positive", id="positive word label", style={"color": "white", "margin-left": "45%"}),
                             html.Img(id="w_positive_cloud", style={"width":"80%"} , src=""),
                             html.Label("Negative", id="positive word label",style={"color": "white", "margin-left": "45%"}),
                             html.Img(id="w_negative_cloud", style={"width":"80%"} , src="")
                         ]),
                html.Div(id="w_neutral_clouds_cont", className="subcont",
                         style={"width": "50%", "height": "100%", "padding": "5%", "dispaly":"flex", "flex-direction":"column"},
                         children=[
                             html.Label("Neutral", id="positive word label",
                                        style={"color": "white","margin-left": "25%"}),
                             html.Img(id="w_neutral_cloud", style={"width": "80%"}, src="")
                         ]),
            ]),

            html.Label("Sentences", id="sent label", style={"color": "white", "margin-left": "45%"}),
            html.Div(id="s_clouds_cont", className="subcont",
                     style={"display": "flex", "flex-direction": "row", "height": "50%", "width": "100%",  "float":"bottom"}, children=[
                    html.Div(id="s_posneg_clouds_cont", className="subcont",
                             style={"display": "flex", "flex-direction": "column","width": "50%", "height": "50%","padding": "5%" },
                             children=[
                                 html.Label("Positive", id="positive sent label",style={"color": "white", "margin-left": "45%"}),
                                 html.Img(id="s_positive_cloud", style={"width": "80%"}, src=""),
                                 html.Label("Negative", id="positive sent label",style={"color": "white", "margin-left": "45%"}),
                                 html.Img(id="s_negative_cloud", style={ "width": "80%"}, src="")
                             ]),
                    html.Div(id="s_neutral_clouds_cont", className="subcont",
                             style={"width": "50%", "height": "100%", "dispaly":"flex", "flex-direction":"column","padding": "5%"},
                             children=[
                                 html.Label("Neutral", id="positive sent label",
                                            style={"color": "white","margin-left": "25%"}),
                                 html.Img(id="s_neutral_cloud", style={"width": "80%"}, src="")
                             ]),
                ]),

        ])



    ]),

])

#_______________________________________________________________________________________________________________
#CALLBACKS
@app.callback([Output(component_id="marginal_btn", component_property="disabled")],
              [Input(component_id="marginal_btn", component_property="n_clicks")],
              prevent_initial_call=True)
def update_marginal_btn_state(disabled):
    return [True]

@app.callback(
    [Output(component_id="generate_clouds_btn", component_property="disabled", allow_duplicate=True),
     ],
    [Input(component_id="generate_clouds_btn", component_property="n_clicks")],
    prevent_initial_call=True,

)
def update_gen_cloud_btn_state(disabled):
    return[True]


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

@app.callback(
    [Output(component_id="chor_g", component_property="figure"),
     Output(component_id="general_graphs_cont", component_property="style"),
     Output(component_id="clouds_cont", component_property="style"),
     Output(component_id="clouds_cont", component_property="hidden")
     ],
    [Input(component_id="band_name_slctr", component_property="value"),
     Input(component_id="genre_slctr", component_property="value"),
     Input(component_id="country_slctr", component_property="value"),
     Input(component_id="album_title_slctr", component_property="value"),
     Input(component_id="song_title_slctr", component_property="value"),
     Input(component_id="song_lyric_slctr", component_property="value"),
     Input(component_id="album_year_slctr", component_property="value"),
     Input(component_id="activity_slctr", component_property="value"),
    Input(component_id="song_len_slctr", component_property="value")

     ],
)
def update_map(band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
               album_year_f, activity_f, song_len_f):
    filtered_df=filter_data(df, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                            album_year_f,activity_f, song_len_f)
    return [create_map(filtered_df),{"width":"100%", "height":"100%"},{"width":"0%", "height":"auto"},True]


@app.callback(
    [Output(component_id="loading-output_metacont", component_property="children"),
     Output(component_id="gnatt_g", component_property="figure"),],
    [ Input(component_id="graphs_metacont", component_property="id"),
     Input(component_id="band_name_slctr", component_property="value"),
     Input(component_id="genre_slctr", component_property="value"),
     Input(component_id="country_slctr", component_property="value"),
     Input(component_id="album_title_slctr", component_property="value"),
     Input(component_id="song_title_slctr", component_property="value"),
     Input(component_id="song_lyric_slctr", component_property="value"),
    Input(component_id="album_year_slctr", component_property="value"),
    Input(component_id="activity_slctr", component_property="value"),
Input(component_id="song_len_slctr", component_property="value")

     ]
)
def update_gnatt(loading_val,band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                 album_year_f,activity_f, song_len_f):
    filtered_df = filter_data(df, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                              album_year_f, activity_f, song_len_f)

    return [loading_val, create_gnatt(filtered_df)]




@app.callback(
    [Output(component_id="heats_g", component_property="figure")],
    [Input(component_id="band_name_slctr", component_property="value"),
     Input(component_id="genre_slctr", component_property="value"),
    Input(component_id="country_slctr", component_property="value"),
    Input(component_id="album_title_slctr", component_property="value"),
     Input(component_id="song_title_slctr", component_property="value"),
     Input(component_id="song_lyric_slctr", component_property="value"),
    Input(component_id="album_year_slctr", component_property="value"),
    Input(component_id="activity_slctr", component_property="value"),
Input(component_id="song_len_slctr", component_property="value")


     ]
)
def update_heats(band_names_f,genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                 album_year_f,activity_f, song_len_f):
    filtered_df = filter_data(df, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                              album_year_f, activity_f, song_len_f)
    return [create_heats(filtered_df)]

@app.callback([Output(component_id="marginal_x_cont", component_property="hidden"),
               Output(component_id="marginal_y_cont", component_property="hidden"),
               Output(component_id="gnatt_marginal_x", component_property="figure"),
               Output(component_id="gnatt_marginal_y", component_property="figure"),
               Output(component_id="marginal_btn", component_property="children"),
               Output(component_id="marginal_btn", component_property="disabled",allow_duplicate=True),
               Output(component_id="loading-output_marginal", component_property="children")
               ],
              [Input(component_id="marginal_btn", component_property="n_clicks"),
               Input(component_id="marginal_btn", component_property="id"),
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

              prevent_initial_call=True
)
def update_marginals(n_clicks, loading_val, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                     album_year_f, activity_f, song_len_f):

    if list(dash.callback_context.triggered_prop_ids.values())==["marginal_btn"]:
        if n_clicks%2!=0:
            filtered_df = filter_data(df, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                                      album_year_f, activity_f, song_len_f)
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
    filtered_df = filter_data(df, band_names_f, genre_slctr_f, country_f, album_title_f, song_title_f, song_lyric_f,
                              album_year_f, activity_f, song_len_f)
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
            True,
            False,
            loading_val
           ]
    print("clicked")
    filtered_data=process_text(filtered_df,"all")
    print("finished process")
    img1=create_neutral_word_cloud(filtered_df,filtered_data)
    img1.save("./temp imgs/neutral words cloud.png",dpi=(300, 300))
    img2=create_positive_word_cloud(filtered_df,filtered_data)
    img2.save("./temp imgs/positive words cloud.png",dpi=(300, 300))
    img3=create_negative_word_cloud(filtered_df,filtered_data)
    img3.save("./temp imgs/negative words cloud.png",dpi=(300, 300))

    img4=create_neutral_sent_cloud(filtered_df,filtered_data)
    img4.save("./temp imgs/neutral sents cloud.png",dpi=(300, 300))
    img5=create_positive_sent_cloud(filtered_df,filtered_data)
    img5.save("./temp imgs/positive sents cloud.png",dpi=(300, 300))
    img6=create_negative_sent_cloud(filtered_df,filtered_data)
    img6.save("./temp imgs/negative sents cloud.png",dpi=(300, 300))



    is_busy=False
    return[Image.open("./temp imgs/neutral words cloud.png"),
           Image.open("./temp imgs/positive words cloud.png"),
           Image.open("./temp imgs/negative words cloud.png"),
           Image.open("./temp imgs/neutral sents cloud.png"),
           Image.open("./temp imgs/positive sents cloud.png"),
           Image.open("./temp imgs/negative sents cloud.png"),

           {"width":"60%", "height":"100%"},
           {"width":"40%", "height":"auto"},
           False,
           True, #True if want to block button AFTER clouds on screen
           loading_val
           ]






if __name__=="__main__":
    app.run_server(debug=False)