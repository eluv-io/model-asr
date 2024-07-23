import requests
import argparse

def main():
    files = [
        ('files', ('', open('1.aac', 'rb'))),
        ('files', ('', open('2.aac', 'rb')))
    ]

    response = requests.post(f'http://127.0.0.1:{args.port}/run', files=files)
    print(response.json())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    main()