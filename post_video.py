import requests

if __name__ == "__main__":
    files = {'file': ("test.mp4", open(r'/mnt/2t/home/zhengbowen/jobs/web_api/data/test.mp4', 'rb'))}
    response = requests.post("http://127.0.0.1:5000/getmp4/", data=None, files=files, verify=False, stream=True)
    print(response.status_code)
    print(response.text)

