"""Confirm deletable transport HTML's scientific entries are in tracked inventory."""
import hashlib
import html
import json
import re
from inventory import ROOT,OUT

inventory=json.loads((ROOT/'reports/damage_p1a/20261025/official_archive_inventory.json').read_text(encoding='utf-8'))
pairs=[('1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b',inventory['root_entries'])]
pairs.extend((x['folder']['id'],x['entries']) for x in inventory['splits'].values())
proof=[]
for ident,expected in pairs:
    p=ROOT/'artifacts/damage_p1a/source'/(ident+'.html')
    raw=p.read_text(encoding='utf-8');actual=[]
    for tag in re.findall(r'<tr\b.*?</tr>',raw,re.S):
        a=re.search(r'data-id="([^"]+)"',tag);b=re.search(r'<strong[^>]*>(.*?)</strong>',tag,re.S)
        if a and b:actual.append({'id':a[1],'name':html.unescape(re.sub('<[^>]*>','',b[1])),
                                  'description':html.unescape(re.sub('<[^>]*>',' ',tag))})
    assert sorted(actual,key=lambda x:x['id'])==sorted(expected,key=lambda x:x['id']),(ident,actual)
    proof.append({'cache':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'folder_entries_match_committed_inventory':True,'names_ids_and_full_descriptions_match':True,'entries':len(actual)})
(OUT/'html_cache_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print('HTML directory fields already in committed inventory PASS')
