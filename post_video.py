import requests

if __name__ == "__main__":
    files = {'file': ("test.mp4", open(r'/mnt/2t/home/datasets/TB/test.mp4', 'rb'))}
    response = requests.post("http://127.0.0.1:5001/action/", data=None, files=files, verify=False, stream=True)
    print(response.status_code)
    print(response.text)

