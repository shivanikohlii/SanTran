
import json, urllib.request, csv, re

def readurl(url, **kwargs):
    return list(urllib.request.urlopen(url, **kwargs))[0]

def search_dict_by(term, search_value, dict):
    return dict[list(filter(lambda k: dict[k][term] == value, dict))[0]]
    
def get_index_data():
    """Return a dictionary mapping book names to {book id, name, chapter_count}."""

    json_contents = json.loads(readurl('http://sanskritbible.in/assets/php/scan-index.php'))
    books = json_contents[1]
    # map a book name to its number and number of books
    books = dict(map(lambda book_json: (book_json["0"], {"book_id": book_json["0"], "name": book_json["1"], "chapter_count": book_json["3"]}), books))
    # only gospels are translated so far
    # matthew = search_dict_by("name", "Matthew", books)
    # acts = search_dict_by("name", "Acts", books)
    # books = dict(map(lambda k: (k, books[k]), filter(lambda k: books[k]["book_id"] >= matthew["book_id"] and books[k]["book_id"] <= acts["book_id"], books)))
    
    return books
    
    
def get_chapter_data(book_id, chapter):
    """Return a dictionary mapping data id's to {data_id, book_id, chapter, verse, sanskrit, english}"""
    
    request_data = {
        "BookNo": str(book_id),
        "ChapterNo": str(chapter)
    }
    json_contents = json.loads(readurl('http://sanskritbible.in/assets/php/read-btxt.php', data = bytes(json.dumps(request_data), 'utf-8')))
    if json_contents[0] is None:
        return {}
    
    sanskrit_data = json_contents[0]
    english_data = json_contents[4]
    
    data = dict(map(lambda sanskrit_json: (sanskrit_json["0"], {
        "data_id": sanskrit_json["0"],
        "book_id": sanskrit_json["1"],
        "chapter": sanskrit_json["2"],
        "verse": sanskrit_json["3"], 
        "sanskrit": sanskrit_json["4"].encode('unicode_escape').decode('utf-8')
    }), sanskrit_data))
    for english_json in english_data:
        data[english_json["0"]]["english"] = english_json["4"]
        
    return data
    
def get_all_data():
    """Return a unified version of get_chapter_data for all available chapters.
    (takes a while to run - progress printed to stdout)"""

    index_data = get_index_data()
    all_data = {}

    for book_id in index_data:
        book = index_data[book_id]
        for chapter in range(1, book["chapter_count"]):
            print('{} {}'.format(book["name"], chapter))
            data = get_chapter_data(book_id, chapter)
            if data == {}:
                print('Empty')
                break # empty chapter, no more data in this book probably
            all_data.update(get_chapter_data(book_id, chapter))
    
    return all_data
    
    
def write_data_to_csv(data, file):
    """Write a dictionary of object dictionaries (such as those from get_all_data) to a csv file."""
    data_cols = list(data[list(data)[0]])
    rows = [data_cols] + [[data[data_key][col] for col in data_cols] for data_key in data]

    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
def convert_sanskrit(uni):
    a = bytearray(uni, encoding = "utf-8").decode('unicode-escape')
    return a
        
def split_data_into_sentence_files(csv_file, out_sentence_file_prefix, en_col=5, san_col=4, convert_san=False):
    rows = []
    if convert_san:
        with open(csv_file, 'rb') as f:
            rows = list(f)
        csv_file = 'decoded-' + csv_file
        with open(csv_file, 'wb') as f:
            for row in rows:
                f.write(bytearray(row.decode('utf-8'), encoding='unicode-escape').replace(b'\\r\\n', b''))
                f.write(b'\n')
        
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    with open(out_sentence_file_prefix + '.en', 'w') as f:
        for row in rows:
            en = row[en_col].strip()
            en = re.sub('([^\s]+?)([,:;"\'\\[\\]?\\.!@#\$%\^&*\(\)`\|]+?)', '\\1 \\2', en)
            en = re.sub('([,:;"\'\\[\\]?\\.!@#\$%\^&*\(\)`\|]+?)([^\s]+?)', '\\1 \\2', en)
            f.write(en)
            f.write('\n')
    with open(out_sentence_file_prefix + '.san', 'w') as f:
        for row in rows:
            en = row[san_col].strip()
            en = re.sub('([^\s]+?)([,:;"\'\\[\\]?\\.!@#\$%\^&*\(\)`\|]+?)', '\\1 \\2', en)
            en = re.sub('([,:;"\'\\[\\]?\\.!@#\$%\^&*\(\)`\|]+?)([^\s]+?)', '\\1 \\2', en)
            f.write(en)
            f.write('\n')
    
if __name__ == '__main__':
    index_data = get_index_data()
    write_data_to_csv(index_data, 'books.csv')
    data = get_all_data()
    write_data_to_csv(data, 'data.csv')