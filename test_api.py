import requests

if __name__ == "__main__":
    files = {'file': ("test.mp4", open(r'/mnt/2t/home/datasets/TB/test.mp4', 'rb'))}
    response = requests.post("http://127.0.0.1:5000/action/", data=None, files=files, verify=False, stream=True)
    # files = {'file': ("test_lintao2.mp4", open(r'/mnt/2t/home/zhengbowen/person_search_demo-master/data/samples/test_lintao2.mp4', 'rb'))}
    # response = requests.post("http://127.0.0.1:5000/reid/", data=None, files=files, verify=False, stream=True)
    print(response.status_code)
    print(response.text)

