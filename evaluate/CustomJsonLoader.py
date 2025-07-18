import json
def _read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
#读取的json文件是一行dic
def _read_json_by_lines(file_path):
    filtered_qrels = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a separate JSON object
            try:
                json_object = json.loads(line)
                filtered_qrels.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
    return filtered_qrels

def read_json(file_path):
    try:
        return _read_json(file_path)
    except:
        return _read_json_by_lines(file_path)
def save_json(file_path, json_data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent = 2, ensure_ascii=False)
def list2dict(data):
    dict_data = {}
    for item in data:
        for k,v in item.items():
            dict_data[k] = v
    return dict_data
