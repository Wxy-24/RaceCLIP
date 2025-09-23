# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pickle
import pandas as pd
import requests
from lxml.html import fromstring
from cachetools import cached, TTLCache
import umls_api


abbr={'RQ':'is related and possibly synonymous with',
      'SY':'is asserted to be synonymy of',
      'RN':'denotes a narrower concept compared with',
      'RB':'denotes a broader concept compared with',
      'QB':'can be qualified by',
      'PAR':'include',
      'CHD':'is part of (belongs to)',    
      }


TTL_7HRS = TTLCache(maxsize=2, ttl=25200)

def cui2name(CUI):
    res = umls_api.API(api_key='72d06e11-2fa4-4bf1-b702-ee2d852038a7').get_cui(cui=CUI)
    return res['result']['name']
def link2name(link):
    res = umls_api.API(api_key='72d06e11-2fa4-4bf1-b702-ee2d852038a7').get_snomedct(link=link)
    return res['result']['name']

class Auth:
    def __init__(self, api_key):
        self._api_key = api_key

    @cached(TTL_7HRS)
    def get_single_use_service_ticket(self):
        url = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Accept': 'text/plain',
            'User-Agent': 'python'
        }
        resp = requests.post(
            url, data={'apikey': self._api_key}, headers=headers
        )
        resp.raise_for_status()
        html = fromstring(resp.text)
        ticket_granting_ticket_url = html.xpath('//form/@action')[0]

        resp = requests.post(
            ticket_granting_ticket_url,
            data={'service': 'http://umlsks.nlm.nih.gov'},
            headers=headers
        )
        resp.raise_for_status()
        single_use_service_ticket = resp.text
        return single_use_service_ticket

class API:
    BASE_URL = 'https://uts-ws.nlm.nih.gov/rest'
                # https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0155502/definitions?apiKey=YOUR_APIKEY

    def __init__(self, *, api_key, version='current'):
        self._auth = Auth(api_key=api_key)
        self._version = version
        self.api_key=api_key
        self.foreign=['SCTSPA','MSHPOR','MSHSPA','MSHCZE','MSHSWE','MSHNOR']

    def get_cui(self, cui):
        url = f'{self.BASE_URL}/content/{self._version}/CUI/{cui}'
        return self._get(url=url)
    
    def get_name(self, cui):
        url = f'{self.BASE_URL}/content/{self._version}/CUI/{cui}'
        return self._get(url=url)['result']['name']
    
    def get_def(self, cui):
        defi=[]
        url = f'{self.BASE_URL}/content/{self._version}/CUI/{cui}/definitions?apiKey='+self.api_key
        try:
            ret = self._get(url=url)['result']
            for source in ret:
                if source['rootSource'] not in self.foreign:
                    defi.append(source['value'])
            res=defi[0]
            for r in defi:
                if len(r)< len(res):
                    res=r
            return res
        except:
            return 'HTTPError'
        
    def get_defall(self, cui):
        url = f'{self.BASE_URL}/content/{self._version}/CUI/{cui}/definitions?apiKey='+self.api_key
        return self._get(url=url)['result']
    def get_rel(self, cui):
        url = f'{self.BASE_URL}/content/{self._version}/CUI/{cui}/relations?apiKey='+self.api_key
        total= self._get(url=url)['result']
        r=[]
        print('Query CUI:',cui)
        for info in total:
            source=info['rootSource']
            if len(source)!=6 or source=='MEDCIN':
                if source=='KCD5':
                    continue
                try:
                    head=info['relatedFromIdName'] 
                except KeyError:
                    # head='XXX '
                    head=cui2name(cui)
                try:
                # if hasattr(info, "relatedIdName"):
                    tail=info["relatedIdName"]
                    print('relatedId:',tail)
                except KeyError:
                    print('relatedId:',info["relatedId"])
                    if info["relatedId"].split('/')[-2]=='CUI':
                        tail=cui2name(info["relatedId"].split('/')[-1])
                    else:
                        tail=link2name(info["relatedId"])
                try:
                    rel_label=info['additionalRelationLabel']
                    if rel_label=="":
                        rel_label=info['relationLabel']
                    print('RelationLabel:',rel_label)
                except KeyError:
                    print('RelationLabel:',info['relationLabel'])
                    rel_label=info['relationLabel']
                if rel_label=='RO':
                    continue
                if rel_label in abbr.keys():
                    rel_label=abbr[rel_label]
                if rel_label=="inverse_isa":
                    rel=' '.join([tail,' is a ',head])
                else:
                    rel=' '.join([head,rel_label,tail])

                r.append([rel,source])
        return r

            
    
    def get_cui_code(self, keyword):
        url = f'{self.BASE_URL}/search/{self._version}/?string={keyword}'
        candidates= self._get(url=url)['result']['results']
        for i in range(3):
            cui=candidates[i]['ui']
            name=candidates[i]['name']
            print(f'No.{i+1} => CUI = {cui}, Concept Name: {name}')
        return candidates[0]['ui']

    def _get(self, url):
        ticket = self._auth.get_single_use_service_ticket()
        resp = requests.get(url, params={'ticket': ticket})
        resp.raise_for_status()
        return resp.json()
    
    
# cui_rel = API(api_key='72d06e11-2fa4-4bf1-b702-ee2d852038a7').get_rel(cui='C2712342')
# print(cui_rel)
    

# cui_def = API(api_key='72d06e11-2fa4-4bf1-b702-ee2d852038a7').get_def(cui='C2712342')
# print(cui_def)

# print(cui2name('C2712342'))
# https://uts-ws.nlm.nih.gov/rest/content/2024AB/source/SNOMEDCT_US/836293000?apiKey=72d06e11-2fa4-4bf1-b702-ee2d852038a7

    


CUI=[]
for split in ['train','valid']:
    csv = pd.read_csv(f'/home/wxy/桌面/NEW-ROCO-{split}.csv')
    c_list=csv['C'].tolist()
    for x in c_list:
        mid=x[2:-2]
        # print(mid)
        ele=mid.split("', '")
        for cui in ele:
            print(cui)
            CUI.append(cui)
CUI=list(set(CUI)) 
print('len(CUI):',len(CUI))
CUI.remove('C1134719')
CUI.remove('C0475380')
CUI.append('C2076528')
CUI.append('C0474781')
# print(CUI)
   
cui,relation,source=[],[],[]
for idx,c in enumerate(CUI):
    cui_rel = API(api_key='72d06e11-2fa4-4bf1-b702-ee2d852038a7').get_rel(cui=c)
    for rel,s in cui_rel:
        print(idx,c,rel,s)
        cui.append(c)
        relation.append(rel)
        source.append(s)

x = {'CUI': cui, 'relation': relation, 'source':source} 
rela = pd.DataFrame(x)
rela.to_csv('/home/wxy/NewROCO_CUIrel_new_0401.csv', sep=';', index=False)


img2rel={}
for split in ['train','valid']:
    csv = pd.read_csv(f'/home/wxy/NewROCO_{split}_RAGrel.csv',sep=';')
    img_list=csv['img'].tolist()
    rel_list=csv['relation'].tolist()
    for idx,img in enumerate(img_list):
        r=rel_list[idx][2:-2].split("', '")
        img2rel[img]=r
        print(idx,img,len(r))


print(len(img2rel.keys()))
with open('/home/wxy/img2rel_NewROCO.pkl', 'wb') as f:
    pickle.dump(img2rel,f)


df=pd.read_csv('/home/wxy/NewROCO_CUIrel_new_0401_refine.csv',sep=';')
CUI=df['CUI'].tolist()
REL=df['relation'].tolist()
ele=[]
rel_dict={}
l=len(CUI)
previous='C0301559'
final_cui,final_rel=[],[]
for i in range(l):
    if CUI[i]==previous:
        ele.append(REL[i])
        if i==l-1:
            rel_dict[CUI[i]]=ele
            final_cui.append(CUI[i])
            final_rel.append(ele)
    else:
        rel_dict[CUI[i]]=ele
        final_cui.append(CUI[i-1])
        final_rel.append(ele)

        ele=[REL[i]]
        previous=CUI[i]

rela = pd.DataFrame( {'CUI': final_cui, 'relation': final_rel} ) 
rela.to_csv('/home/wxy/NewROCO_FinalCUIrel.csv', sep=';', index=False)

split=['train','valid']
for s in split:
    df=pd.read_csv(f'/home/wxy/NEW-ROCO-{s}.csv',sep=',')
    img=df['A'].tolist()
    cap=df['B'].tolist()
    CUI=df['C'].tolist()
    img_rel=[]
    l=len(img)
    for i in range(l):
        sub=[]
        c_list=CUI[i][2:-2].split("', '")
        print(c_list)
        for c in c_list:
            try:
                ele_rel=rel_dict[c]
            except:
                continue
            for r in ele_rel:
                sub.append(r)
        img_rel.append(sub)

    rela = pd.DataFrame( {'CUI': img,'caption': cap, 'relation': img_rel} ) 
    rela.to_csv(f'/home/wxy/NewROCO_{s}_IMGrel.csv', sep=';', index=False)


# with open('/home/wxy/CUI2def_NewROCO.pkl', 'rb') as f:
#     cui2d=pickle.load(f)



# res=list(set(res))
# wr=[]
# for r in res:
#     print(r)
#     wr.append(r+'\n')
# print(len(res))
# with open('/home/wxy/桌面/ABBR_REL','w') as f:
#     f.writelines(wr)  


# ff=open('/home/wxy/桌面/ABBR_REL_ISA')
# isa=ff.readlines()
# for i in isa:
#     i=i.strip('\n')
# print(isa)
