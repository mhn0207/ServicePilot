"""Run Docker DNS regressions outside pytest with python -X utf8 tests/verify_gateway_recovery.py."""

import argparse
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"{sys.argv[1]} {self.path}".encode())

ThreadingHTTPServer(("", 8000), Handler).serve_forever()
"""
PREFIXES = ('/api/servicepilot/health', '/api/zhiying/health')
MODES = (
    ('backend', 'nginx:alpine', ROOT/'config/nginx/nginx.conf', 'nginx.conf', 'zhiying',
     ('/health', '/mcp/')),
    ('unified', 'nginx:alpine', ROOT/'config/nginx/web-gateway.conf', 'nginx.conf', 'zhiying',
     (*PREFIXES, '/health', '/mcp/')),
    ('direct-web', 'nginx:1.27-alpine', ROOT/'config/nginx/frontend.conf', 'conf.d/default.conf', 'zhiying',
     (*PREFIXES, '/openapi.json')),
    ('standalone-web', 'nginx:1.27-alpine', ROOT.parent/'frontend/docker/nginx-gateway.conf',
     'conf.d/default.conf', 'zhiying-python', PREFIXES),
    ('standalone-direct-web', 'nginx:1.27-alpine', ROOT.parent/'frontend/docker/nginx.conf',
     'conf.d/default.conf', 'zhiying-python', (*PREFIXES, '/openapi.json')),
)


def docker(*args, check=True):
    return subprocess.run(['docker', *args], check=check, text=True, encoding='utf-8',
                          stdout=subprocess.PIPE).stdout.strip()


def verify(mode, image, config, destination, alias, paths, backend_image):
    network = f'sp-dns-{uuid4().hex[:8]}'
    with ExitStack() as stack:
        docker('network', 'create', network)
        stack.callback(docker, 'network', 'rm', network, check=False)

        def start(suffix, *options):
            name = f'{network}-{suffix}'
            docker('run', '-d', '--network', network, '--name', name, *options)
            stack.callback(docker, 'rm', '-fv', name, check=False)
            return name

        old = start('old', '--network-alias', alias, backend_image, 'python', '-c', UPSTREAM, 'old')
        gateway = start('nginx', '-p', '127.0.0.1::80', '-v',
                        f'{config}:/etc/nginx/{destination}:ro', image)
        address = docker('port', gateway, '80/tcp')
        started = docker('inspect', '--format', '{{.State.StartedAt}}', gateway)
        with httpx.Client(base_url='http://' + address, trust_env=False, timeout=2) as client:
            def wait_for(marker):
                deadline = time.monotonic() + 18
                while time.monotonic() < deadline:
                    try:
                        replies = [client.get(path, params={'probe': 'routing'}).text for path in paths]
                        expected = [f"{marker} {path.removeprefix('/api/servicepilot').removeprefix('/api/zhiying')}?probe=routing"
                                    for path in paths]
                        if replies == expected:
                            return
                    except httpx.RequestError:
                        pass
                    time.sleep(1)
                raise AssertionError(f'{mode}: gateway did not forward every path to {marker}')

            wait_for('old')
            new = start('new', '--network-alias', alias, backend_image, 'python', '-c', UPSTREAM, 'new')
            ip_format = '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
            assert docker('inspect', '--format', ip_format, old) != docker('inspect', '--format', ip_format, new)
            docker('stop', old)
            wait_for('new')
            assert docker('inspect', '--format', '{{.State.StartedAt}}', gateway) == started
        print(f'PASS {mode}: changed upstream IP recovered with the original gateway', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-image', default='python:3.12-slim', help='Docker image containing Python')
    args = parser.parse_args()
    for mode in MODES:
        verify(*mode, args.backend_image)
