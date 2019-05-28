#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path as osp
import re
import shutil
import sys
import tempfile

import requests
import six
import tqdm


# BORROWED FROM GDOWN



CHUNK_SIZE = 512 * 1024  # 512KB


def get_url_from_gdrive_confirmation(contents):
    url = ''
    for line in contents.splitlines():
        m = re.search('href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = 'https://docs.google.com' + m.groups()[0]
            url = url.replace('&amp;', '&')
            return url
        m = re.search('confirm=([^;&]+)', line)
        if m:
            confirm = m.groups()[0]
            url = re.sub(r'confirm=([^;&]+)', r'confirm='+confirm, url)
            return url
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace('\\u003d', '=')
            url = url.replace('\\u0026', '&')
            return url


def is_google_drive_url(url):
    m = re.match('^https?://drive.google.com/uc\?id=.*$', url)
    return m is not None


def download(url, output, quiet):
    url_origin = url
    sess = requests.session()

    is_gdrive = is_google_drive_url(url)

    while True:
        res = sess.get(url, stream=True)
        if 'Content-Disposition' in res.headers:
            # This is the file
            break
        if not is_gdrive:
            break

        # Need to redirect with confiramtion
        url = get_url_from_gdrive_confirmation(res.text)

        if url is None:
            print('Permission denied: %s' % url_origin, file=sys.stderr)
            print("Maybe you need to change permission over "
                  "'Anyone with the link'?", file=sys.stderr)
            return

    if output is None:
        if is_gdrive:
            m = re.search('filename="(.*)"',
                          res.headers['Content-Disposition'])
            output = m.groups()[0]
        else:
            output = osp.basename(url)

    output_is_path = isinstance(output, six.string_types)

    if not quiet:
        print('Downloading...', file=sys.stderr)
        print('From:', url_origin, file=sys.stderr)
        print('To:', osp.abspath(output) if output_is_path else output,
              file=sys.stderr)

    if output_is_path:
        tmp_file = tempfile.mktemp(
            suffix=tempfile.template,
            prefix=osp.basename(output),
            dir=osp.dirname(output),
        )
        f = open(tmp_file, 'wb')
    else:
        tmp_file = None
        f = output

    try:
        total = res.headers.get('Content-Length')
        if total is not None:
            total = int(total)
        if not quiet:
            pbar = tqdm.tqdm(total=total, unit='B', unit_scale=True)
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            if not quiet:
                pbar.update(len(chunk))
        if not quiet:
            pbar.close()
        if tmp_file:
            f.close()
            shutil.copy(tmp_file, output)
    except IOError as e:
        print(e, file=sys.stderr)
        return
    finally:
        try:
            if tmp_file:
                os.remove(tmp_file)
        except OSError:
            pass

    return output



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'url_or_id', help='url or file id (with --id) to download file from')
    parser.add_argument('-O', '--output', help='output filename')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress standard output')
    parser.add_argument('--id', action='store_true',
                        help='flag to specify file id instead of url')
    args = parser.parse_args()

    print(args)
    if args.output == '-':
        if six.PY3:
            args.output = sys.stdout.buffer
        else:
            args.output = sys.stdout

    download(args.url_or_id, args.output, args.quiet)

if __name__ == '__main__':
    main()
