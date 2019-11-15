#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:32:39 2019

@author: github.com/sahandv
"""

import sys, os, time
import platform
from pathlib import Path


def scopus_initialize(ignore_py_version = False,
                      ignore_scopus_config = False):
    
    if platform.system() != 'Linux':
        print("\n * It is recommended to use a Linux system.")
        
    if ignore_py_version is False:
        try:
            assert sys.version_info >= (3, 5)
        except AssertionError:
            sys.exit("\n * Please use Python 3.5 +")
    
    try:
        import mmap
    except ImportError:
        sys.exit("\n * Please install mmap.")

    try:
        import tqdm
    except ImportError:
        sys.exit("\n * Please install tqdm.")
    
    try:
        import pandas
    except ImportError:
        sys.exit("\n * Please install pandas.")
    
    try:
        import nltk
    except ImportError:
        sys.exit("\n * Please install nltk.")
    
    
    if ignore_scopus_config is False:
        try:
            import scopus
        except ImportError:
            sys.exit("\n * Please install scopus package before using this code. Try usng 'pip install scopus'.")
        my_file = Path(str(Path.home())+'/.scopus/config.ini')
        if my_file.is_file():
            print("\n * Configuration file already exists at "+str(Path.home())+'/.scopus/config.ini'+". You may the file and edit the entries manually.")
        else:
            scopus.utils.create_config()
    


def search_scopus(query, download = True):
    from scopus import ScopusSearch
    result = ScopusSearch(query,download = download)
    return result


def retrieve_abstract_try(eid,view = 'REF',param = 'references'):
    from scopus import AbstractRetrieval
    try:
        refs = AbstractRetrieval(eid, view = view)._json[param]
    except KeyError:
        print('An error occurred (1) ...')
        return 1
    except UnboundLocalError:
        print('An error occurred (2). Probably an empty eID provided? ')
        return 2
    except KeyboardInterrupt:
        sys.exit("Interrupting due to user command.")
    except:
        print('An error occurred (?)...')
        return 0
    else:
        return refs
        

# =============================================================================
# Get references for document by eID
#        
#    eid : the publication scopus eid
#    retry : the number of retries if it fails due to some reason like quota
#    force : don't prompt to fail the program and go more than retry limit
# =============================================================================
def get_refs_by_eid(eid, retry = 1, force = False):   
#    from scopus import AbstractRetrieval
    import pandas as pd
    refs = None
    for i in range(retry+1):
        refs = retrieve_abstract_try(eid)
        
        if refs == 1 or refs == 2 or (refs == 0 and retry == 0):
            print('Returning None.')
            return None, None, None
        
        if refs == 0:
            print("Trying again: " , i ," of ", retry)
            if i >= retry and force is False:
                print("How many more retries do you want? [0 means break] ")
                input_val = input()
                input_val = int(input_val)
                if input_val == 0:
                    print('Returning None.')
                    return None, None, None
                else:
                    get_refs_by_eid(eid, input_val, force)
                    
        if refs != 1 and refs != 2 and refs != 0:
            break
        

    try:
        ref_list_full = refs['reference']
        ref_count = refs['@total-references']
    except TypeError:
        print('Returning None.')
        return None, None, None 
    
    if ref_count == '1' or ref_count == 1:
        print('The article has only 1 refrences.')
        ref_list_full = [ref_list_full]
    if ref_count == '0' or ref_count == 0:
        print('The article has 0 refrences! Returning None and ignoring.')
        return None, None, None
    
    ref_list_full_df = pd.DataFrame(ref_list_full)
    
    try:
        ref_list_eid = ref_list_full_df['scopus-eid']
    except KeyError:
        ref_list_eid = None
        
    return ref_list_eid,ref_list_full,ref_count

# =============================================================================
# Fetch references for a series of eIDs
# =============================================================================
def get_refs_from_publications_df(dataframe_eid,verbose = True,retry = 0, force = False):
    import pandas as pd
    import psutil
    memory = psutil.virtual_memory()
    if len(dataframe_eid) > 10000 and memory[1]/1000000000 < 8:
        input("There are too many records to fetch (>10k) for your free memory. Please free up memory or press Enter to continue anyway.")
        
    all_refs = []
    valid_eids = []
    for eid in dataframe_eid:
        if verbose is True:
            print('Fetching references for ',eid)
        refs_eid, _, _, = get_refs_by_eid(eid,retry,force)
        if refs_eid is not None:
            all_refs.append(refs_eid)
            valid_eids.append(eid)
            
    
    return valid_eids,all_refs
    
# =============================================================================
# Get number of lines in file
# =============================================================================
def get_num_lines(file_address):
    import mmap
    fp = open(file_address, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

# =============================================================================
# Read author properties from 'author' field
# =============================================================================
def read_author(author_field,retry_no=0):
    import json
    
    retry_no+=1
    author_json = None
    author_field = author_field.replace("\'", "\"")
    author_field = author_field.replace("None", "\"\"")
    
    try:
        author_json = json.loads(author_field)
        
    except json.JSONDecodeError as decodererror:
        if 'Expecting value' in decodererror.msg:
            author_field = author_field[1:-1]
            read_author(author_field,retry_no)
            
    if retry_no>4:
        return author_json
    
#        if 'Extra data' in decodererror.msg:
#            author_field = author_field[2:-2]
#            author_field = author_field.split('}, {')
#            author_json = []
#            for row in author_field:
#                row = '{'+row+'}'
#                try:
#                    author_json.append(json.loads(row))
#                except json.JSONDecodeError as decodererror:
#                    print(decodererror.args,'\n',row)
            
    return author_json


# =============================================================================
# Turn author dictionary to string in WoS format
# =============================================================================
def authors_to_string(authors_row):
    if authors_row != '':
        strings = []
        for author in authors_row:
            strings.append(author['surname']+', '+author['initials'].replace('.',''))
        return '; '.join(strings)
    else:
        return ''

# =============================================================================
# Turn JSON data to Pandas DataFrame
# =============================================================================
def json_to_df(file_address = None,verbose = True):
    import json
    from tqdm import tqdm
    import pandas as pd
    
    if file_address is None:
        from stat import S_ISREG, ST_CTIME, ST_MODE
        dir_path = str(Path.home())+'/.scopus/scopus_search/COMPLETE/'
        data = (os.path.join(dir_path, fn) for fn in os.listdir(dir_path))
        data = ((os.stat(path), path) for path in data)
        data = ((stat[ST_CTIME], path) 
            for stat, path in data if S_ISREG(stat[ST_MODE]))
        print("\n * No json file was supplied. Searching the scopus cache files...")
        
        for cdate, path in sorted(data):
            print('     - ', time.ctime(cdate), os.path.basename(path))
            file_address = path
        
    if file_address is not None:
        print("Will process ",file_address,'. File size: ',int(os.path.getsize(file_address)/1000000),'MB')
        try:
            input("Press enter to process...")
        except SyntaxError:
            pass
    else:
        sys.exit("Please supply a json file. The cache was also empty. :(")
    
    all_publications = []
    counter = 0
    with open(file_address) as publication_data_text:
        for line in tqdm(publication_data_text, total=get_num_lines(file_address)):
            publication_data_dict = json.loads(line)
            try:
                publication_data_dict["author_keywords"] = publication_data_dict['authkeywords'].split(" | ")
            except KeyError:
                if verbose is True:
                    print(counter," - This publication has no author keywords!")
            
            all_publications.append(publication_data_dict)
            counter+=1
    print("\nFinished parsing.")
    return pd.DataFrame(all_publications)


def find_first_big_csv(filename,chunksize,names,column,needle):
    import pandas as pd
    for chunk in pd.read_csv(filename, chunksize=chunksize,names=names):
        result = chunk[chunk[column].str.contains(needle)]
        if result.shape[0]>0:
            return result
    return False
        
# =============================================================================
# Wos to Scopus format
# =============================================================================
import pandas as pd
import numpy as np
from tqdm import tqdm
import threading
import concurrent.futures
from concurrent import futures

def wos_author_unifyer(wos_data_row):
    value = wos_data_row['BA'] if pd.isnull(wos_data_row['AU']) else wos_data_row['AU']
    value = wos_data_row['CA'] if pd.isnull(value) else value
    value = wos_data_row['GP'] if pd.isnull(value) else value
    value = wos_data_row['BE'] if pd.isnull(value) else value
    return value

def wos_publication_unifyer(wos_data_row):
    value = wos_data_row['CT'] if pd.isnull(wos_data_row['SO']) else wos_data_row['SO']
    value = wos_data_row['SE'] if pd.isnull(value) else value
    return value

def wos_author_full_unifyer(wos_data_row):
    value = wos_data_row['BF'] if pd.isnull(wos_data_row['AF']) else wos_data_row['AF']
    return value

def wos_doi_unifyer(wos_data_row):
    value = wos_data_row['D2'] if pd.isnull(wos_data_row['DI']) else wos_data_row['DI']
    return value

def wos_data_reshape_to_scopus(wos_data,scopus_columns):
    if wos_data.shape[0] > 1000:
        print('Beware that this method will take a very long time to complete considering the size of your data.')
        
    data_reshaped_wos_scopus = pd.DataFrame(np.nan, index=list(range(0,wos_data.shape[0])), columns=scopus_columns) # an empty dataframe with goal shape

    index_cons = 0
    for index, row in tqdm(wos_data.iterrows(), total=wos_data.shape[0]):
        data_reshaped_wos_scopus['author'][index_cons] = wos_author_full_unifyer(row)
        data_reshaped_wos_scopus['author_str'][index_cons] = wos_author_unifyer(row)
        data_reshaped_wos_scopus['authkeywords'][index_cons] = row['DE']
        data_reshaped_wos_scopus['citedby-count'][index_cons] = row['Z9']
    #    data_reshaped_wos_scopus['orcid'][index_cons] = row['OI']
        data_reshaped_wos_scopus['dc:description'][index_cons] = row['AB']
        data_reshaped_wos_scopus['dc:title'][index_cons] = row['TI']
        data_reshaped_wos_scopus['fund-no'][index_cons] = row['FU']
        data_reshaped_wos_scopus['fund-sponsor'][index_cons] = row['FX']
        data_reshaped_wos_scopus['subtypeDescription'][index_cons] = row['DT']
        data_reshaped_wos_scopus['prism:coverDate'][index_cons] = row['PD']
        data_reshaped_wos_scopus['prism:doi'][index_cons] = wos_doi_unifyer(row)
        data_reshaped_wos_scopus['prism:publicationName'][index_cons] = wos_publication_unifyer(row)
        data_reshaped_wos_scopus['prism:isbn'][index_cons] = row['BN']
        data_reshaped_wos_scopus['prism:issn'][index_cons] = row['SN']
        data_reshaped_wos_scopus['prism:eIssn'][index_cons] = row['EI']
        data_reshaped_wos_scopus['prism:volume'][index_cons] = row['VL']
        data_reshaped_wos_scopus['prism:issueIdentifier'][index_cons] = row['IS']
        data_reshaped_wos_scopus['prism:pageRange'][index_cons] = row['BP']+'-'+row['EP'] if (pd.notnull(row['BP']) & pd.notnull(row['EP'])) else None
        data_reshaped_wos_scopus['pubmed-id'][index_cons] = row['PM']
        data_reshaped_wos_scopus['year'][index_cons] = row['PY']
        data_reshaped_wos_scopus['data_source'][index_cons] = index
        index_cons+=1
        
    data_reshaped_wos_scopus['authkeywords'] = data_reshaped_wos_scopus['authkeywords'].astype(str).str.replace(',','|')
    return data_reshaped_wos_scopus

def wos_data_reshape_to_scopus_row(index,row,scopus_columns):
    
    data_reshaped_wos_scopus = pd.DataFrame(np.nan, index=list(range(0,1)), columns=scopus_columns) # an empty dataframe with goal shape

    data_reshaped_wos_scopus['author'][0] = wos_author_full_unifyer(row)
    data_reshaped_wos_scopus['author_str'][0] = wos_author_unifyer(row)
    data_reshaped_wos_scopus['authkeywords'][0] = row['DE']
    data_reshaped_wos_scopus['citedby-count'][0] = row['Z9']
#    data_reshaped_wos_scopus['orcid'][index_cons] = row['OI']
    data_reshaped_wos_scopus['dc:description'][0] = row['AB']
    data_reshaped_wos_scopus['dc:title'][0] = row['TI']
    data_reshaped_wos_scopus['fund-no'][0] = row['FU']
    data_reshaped_wos_scopus['fund-sponsor'][0] = row['FX']
    data_reshaped_wos_scopus['subtypeDescription'][0] = row['DT']
    data_reshaped_wos_scopus['prism:coverDate'][0] = row['PD']
    data_reshaped_wos_scopus['prism:doi'][0] = wos_doi_unifyer(row)
    data_reshaped_wos_scopus['prism:publicationName'][0] = wos_publication_unifyer(row)
    data_reshaped_wos_scopus['prism:isbn'][0] = row['BN']
    data_reshaped_wos_scopus['prism:issn'][0] = row['SN']
    data_reshaped_wos_scopus['prism:eIssn'][0] = row['EI']
    data_reshaped_wos_scopus['prism:volume'][0] = row['VL']
    data_reshaped_wos_scopus['prism:issueIdentifier'][0] = row['IS']
    data_reshaped_wos_scopus['prism:pageRange'][0] = row['BP']+'-'+row['EP'] if (pd.notnull(row['BP']) & pd.notnull(row['EP'])) else None
    data_reshaped_wos_scopus['pubmed-id'][0] = row['PM']
    data_reshaped_wos_scopus['year'][0] = row['PY']
    data_reshaped_wos_scopus['data_source'][0] = index
    data_reshaped_wos_scopus['authkeywords'] = data_reshaped_wos_scopus['authkeywords'].astype(str).str.replace(',','|')
    
    return data_reshaped_wos_scopus


def wos_data_reshape_to_scopus_mp(wos_data,scopus_columns,process=2):
             
    results = []
    jobs = []
    job_counter = 0
    counter_success = 0
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers = process) as executor:
        for index, row in wos_data.iterrows():
            job_counter +=1
            print(job_counter,'reshaping row',index)
    #        jobs.append(executor.submit(search,doi,citing_dois_index,column)) # search pandas
            jobs.append(executor.submit(wos_data_reshape_to_scopus_row,index,row,scopus_columns)) # search numpy
    
    
        for job in futures.as_completed(jobs):
            result_done = job.result()
            if result_done is not None:
                results.append(result_done)
                print('done',counter_success)
                counter_success=counter_success+1
    #            print(counter_success,"found",result_done.iloc[0].name) # search pandas
    
    print((time.time() - start_time),"seconds")

    return results