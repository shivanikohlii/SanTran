# import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import bs4
import unicodedata
import codecs
import string
import re
import csv



def getGita(chap, verse):
    quote_page = "https://www.holy-bhagavad-gita.org/chapter/"+str(chap)+"/verse/" + str(verse)
    req = Request(quote_page, headers={'User-Agent': 'Mozilla/5.0'})
    # query the website and return the html to the variable ‘page’
    try:
        page = urlopen(req).read()
    except Exception as e:
        print("page that failed was: " + quote_page)
        raise e
    
    # parse the html using beautiful soup and store in variable `soup`
    soup = BeautifulSoup(page, 'html.parser')
    
    
    sa = soup.find(id="originalVerse") #finds sanskrit
    pTagSA = sa.p
    pTagSA = [str(elm) for elm in pTagSA.contents if (type(elm) == bs4.element.NavigableString)]#this only keeps strings from the HTML
    #print(pTagSA)
    
    
    SA = ''.join(re.sub(r"\d+\|\|","",str(elem)) for elem in pTagSA) # this removes the | characters found in the sanskrit
    SA = ''.join([i for i in SA if not i.isdigit()]) #idk if this is needed but it takes random verse numbers from the end of the sanskrit
    #print()
    #print(SA)
    
    #-----------------------------------------
    
    en = soup.find(id="translation") #finds the english
    pTagEN = en.p
    #for con in pTagEN.contents:
    #    if type(con) == bs4.element.Tag:
    #        print(str(con.get('id'))+"-"+str(con))

    #pTagEN = [str(elm) for elm in pTagEN.contents if (type(elm) == bs4.element.NavigableString) ] #this only keeps strings from the HTML
    enL = []
    for elm in pTagEN.contents:
        strElm = str(elm)
        if (type(elm) == bs4.element.NavigableString):
            enL.append(strElm)
        if type(elm) == bs4.element.Tag:
            if "<i>" in strElm: 
                strElm = re.sub(r"\</i\>","",strElm)
                strElm = re.sub(r"\<i\>","",strElm)
                enL.append(strElm)

    #print(pTagEN)
    #print()
    EN = ''.join(str(elm).replace('\n','') for elm in enL)
    #EN = pTagEN[1].replace('\n','') #removes newlines from the english
    
    #this gets rid of the punctuation 
    #exclude = set(string.punctuation)                   
    #EN = ''.join(ch for ch in EN if ch not in exclude)
    
    #print(EN)
    mId = "c:"+str(chap)+"v"+str(verse)
    return (mId,SA,EN)

if __name__ == '__main__':
    versesInChap = [46,72,43,42,29,47,30,28,34,42,55,20,35,27,20,24,28,78] 

    #print(getGita(5,3))

    with open('data.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(["id","SA","EN"])
        for i in range(1, 19): #chap
            for j in range(1, versesInChap[i-1]+1): #verse
                print("chap: " +str(i)+"verse: " +str(j))
                csv_out.writerow(getGita(i,j))







