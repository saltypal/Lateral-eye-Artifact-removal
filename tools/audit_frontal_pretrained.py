"""Record public release discovery, without loading EEG or inventing models."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import requests
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if tag == 'a' and name == 'href' and value:
                self.links.append(value)


def inspect_source(item):
    name, url, params = item
    row = {'source': name, 'requested_url': url, 'query': params,
           'checked_utc': datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, params=params, timeout=(10, 30))
        parts = urlsplit(response.url)
        final_url = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
        row.update(http_status=response.status_code, final_url=final_url,
                   response_sha256=hashlib.sha256(response.content).hexdigest())
        if 'perfdrive.com' in parts.netloc or 'captcha' in response.text[:3000].lower():
            row['inspection'] = 'bot challenge; publisher content and supplements unverified'
            return row
        if not response.ok:
            row['inspection'] = 'access failed; no absence claim from this source'
            return row
        if name.startswith('github'):
            data = response.json()
            row.update(total_count=data.get('total_count'), incomplete_results=data.get('incomplete_results'),
                       repositories=[{'name': value['full_name'], 'url': value['html_url']}
                                     for value in data.get('items', [])])
        elif name.startswith('huggingface'):
            row['model_ids'] = [value['id'] for value in response.json()]
        elif name.startswith('crossref'):
            data = response.json()['message']
            row['metadata'] = {key: data.get(key) for key in
                               ['DOI', 'title', 'published', 'published-online',
                                'published-print', 'created', 'link', 'resource', 'relation']}
        elif name.startswith('europepmc'):
            row['records'] = [{key: value.get(key) for key in
                               ['id', 'title', 'isOpenAccess', 'inPMC',
                                'fullTextUrlList', 'dataLinksTagsList']}
                              for value in response.json().get('resultList', {}).get('result', [])]
        elif name.endswith('_xml'):
            root = ET.fromstring(response.content)
            row['publisher_metadata'] = {element.tag.split('}')[-1]: element.text
                for element in root.iter() if element.tag.split('}')[-1] in
                ['doi', 'coverDate', 'openaccess', 'openaccessArticle']}
            row['article_body_available'] = any(element.tag.split('}')[-1] == 'body'
                                                for element in root.iter())
            row['inspection'] = 'publisher XML; article-body availability recorded separately'
        else:
            parser = Links()
            parser.feed(response.text)
            row['release_or_supplement_links'] = sorted({link for link in parser.links
                if any(word in link.lower() for word in
                       ['github', 'huggingface', 'zenodo', 'supplement', 'mmc', 'figshare'])})
            row['inspection'] = 'accessible HTML links only; not proof of full-text access'
    except (requests.RequestException, ValueError, ET.ParseError) as error:
        row.update(inspection='request failed; no absence claim', error_type=type(error).__name__)
    return row


def main():
    checks = []
    for key, phrase in [('wnet', 'EEG-WNet'), ('diffadn', 'Diff-ADN'),
                        ('wnet_compact', 'EEGWNet'), ('diffadn_compact', 'DiffADN')]:
        checks.append(('github_' + key, 'https://api.github.com/search/repositories',
                       {'q': '"' + phrase + '"', 'per_page': 30}))
        checks.append(('huggingface_' + key, 'https://huggingface.co/api/models',
                       {'search': phrase, 'limit': 30}))
    checks.extend([
        ('crossref_wnet', 'https://api.crossref.org/works/10.1016/j.bspc.2026.111553', {}),
        ('crossref_diffadn', 'https://api.crossref.org/works/10.1088/1741-2552/aeab36', {}),
        ('europepmc_diffadn', 'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
         {'query': 'EXT_ID:42772331 AND SRC:MED', 'format': 'json', 'resultType': 'core'}),
        ('publisher_wnet', 'https://www.sciencedirect.com/science/article/pii/S1746809426021117', {}),
        ('publisher_diffadn', 'https://iopscience.iop.org/article/10.1088/1741-2552/aeab36', {}),
        ('publisher_wnet_xml', 'https://api.elsevier.com/content/article/PII:S1746809426021117',
         {'httpAccept': 'text/xml'}),
    ])
    with ThreadPoolExecutor(max_workers=4) as pool:
        evidence = list(pool.map(inspect_source, checks))
    document = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'evidence': evidence,
                'scope': 'Public discovery; failures and empty searches do not prove worldwide absence.',
                'numerical_processing': False}
    target = ROOT / 'docs/provenance/frontal_pretrained_discovery.json'
    target.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    for row in evidence:
        print(row['source'], row.get('http_status', row.get('error_type')),
              row.get('total_count', row.get('model_ids', row.get('inspection', 'metadata recorded'))))


if __name__ == '__main__':
    main()
