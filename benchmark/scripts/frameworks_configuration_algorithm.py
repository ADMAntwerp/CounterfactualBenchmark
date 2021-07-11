import json

fw_config = json.load(open('./benchmark_config.json'))

out_str_order = ''

for fwName, fwConf in fw_config.items():
    out_str_order += f'{fwName},{fwConf["python"]},{fwConf["requirements"]},{fwConf["bechmark_algo_name"]};'

exit(out_str_order)
