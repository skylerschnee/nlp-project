"""
A module for obtaining repo readme and language data from the github API.

Before using this module, read through it, and follow the instructions marked
TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import pandas as pd
import numpy as np

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['Jasonette/JASONETTE-Android',
 'vivchar/RendererRecyclerViewAdapter',
 'Pkmmte/CircularImageView',
 'D-clock/AndroidSystemUiTraining',
 'xujeff/tianti',
 'UCodeUStory/S-MVP',
 'zuiwuyuan/WeChatPswKeyboard',
 'FlyingPumba/SimpleRatingBar',
 'H07000223/FlycoSystemBar',
 'wl9739/BlurredView',
 'matburt/mobileorg-android',
 'ScienJus/spring-restful-authorization',
 '58code/Argo',
 'AzimoLabs/AndroidKeyboardWatcher',
 'square/dagger-intellij-plugin',
 'nvanbenschoten/motion',
 'jpardogo/FlabbyListView',
 'oschina/android-app',
 'quartzjer/TeleHash',
 'hitherejoe/Vineyard',
 'firebase/geofire-java',
 'dupengtao/BubbleTextView',
 'onlylemi/MapView',
 'facebookarchive/proguard',
 'shehabic/Droppy',
 'jblough/Android-Pdf-Viewer-Library',
 'coomar2841/image-chooser-library',
 'RomanTruba/AndroidTouchGallery',
 'AndroidAlliance/EdgeEffectOverride',
 'ikew0ng/Dribbo',
 'f2prateek/progressbutton',
 'HomHomLin/SlidingLayout',
 'square/pollexor',
 'klinker41/android-chips',
 'JustZak/DilatingDotsProgressBar',
 'graalvm/sulong',
 'dibbhatt/kafka-spark-consumer',
 'androidthings/sample-tensorflow-imageclassifier',
 'Sixt/ja-micro',
 'Jude95/Beam',
 'nibnait/algorithms',
 'iammert/ProgressLayout',
 'zccodere/study-imooc',
 'Nukkit/Nukkit',
 'zalando/problem',
 'square/dagger-intellij-plugin',
 'nvanbenschoten/motion',
 'jeasonlzy/HeaderViewPager',
 'jpardogo/FlabbyListView',
 'liuyanggithub/SuperMvp',
 'olahol/reactpack',
 'cbrauckmuller/helium',
 'OverZealous/run-sequence',
 'pavelk2/social-feed',
 'jsmreese/moment-duration-format',
 'ben-eb/gulp-uncss',
 'stolksdorf/Parallaxjs',
 'hxgf/smoke.js',
 'maxzhang/maxzhang.github.com',
 'dylang/grunt-notify',
 'jfairbank/redux-saga-test-plan',
 'mbostock/stack',
 'laravel/elixir',
 'ModusCreateOrg/budgeting',
 'iconic/SVGInjector',
 'olahol/reactpack',
 'cbrauckmuller/helium',
 'leonidas/transparency',
 'FountainJS/generator-fountain-webapp',
 'OverZealous/run-sequence',
 'remoteinterview/compilebox',
 'damonbauer/npm-build-boilerplate',
 'iconic/SVGInjector',
 'olahol/reactpack',
 'cbrauckmuller/helium',
 'auth0-blog/angular2-authentication-sample',
 'OverZealous/run-sequence',
 'koush/electron-chrome',
 'pavelk2/social-feed',
 'jsmreese/moment-duration-format',
 'michaelliao/itranswarp.js',
 'elgris/microservice-app-example',
 'fossasia/Connect-Me',
 'ehynds/jquery-ui-multiselect-widget',
 'guangqiang-liu/OneM',
 'ericjang/tdb',
 'nicolewhite/algebra.js',
 'getify/CAF',
 'PatMartin/Dex',
 'kumailht/responsive-elements',
 'ben-eb/gulp-uncss',
 'Metnew/suicrux',
 'emmenko/redux-react-router-async-example',
 'stolksdorf/Parallaxjs',
 'd0ugal/locache',
 'maxzhang/maxzhang.github.com',
 'dylang/grunt-notify',
 'Ashung/Automate-Sketch',
 'web-perf/react-worker-dom',
 'svrcekmichal/redux-axios-middleware',
 'vfaronov/httpolice',
 'kennethreitz-archive/requests3',
 'Capgemini/Apollo',
 'syrusakbary/pyjade',
 'some-programs/exitwp',
 'slackapi/python-rtmbot',
 'LoyaltyNZ/alchemy-framework',
 'covid19india/api',
 'codekansas/keras-language-modeling',
 'j-bennet/wharfee',
 'csurfer/gitsuggest',
 'aurora95/Keras-FCN',
 'joschu/cgt',
 'espeed/bulbs',
 'cerndb/dist-keras',
 'gornostal/Modific',
 'ParhamP/altify',
 'bitly/asyncmongo',
 'yasintoy/Slack-Gitsin',
 'flashingpumpkin/django-socialregistration',
 'j-bennet/wharfee',
 'kvh/ramp',
 'csurfer/gitsuggest',
 'aurora95/Keras-FCN',
 'bookieio/Bookie',
 'joschu/cgt',
 'TeamHG-Memex/tensorboard_logger',
 'espeed/bulbs',
 'cerndb/dist-keras',
 'gornostal/Modific',
 'hangoutsbot/hangoutsbot',
 'sciyoshi/pyfacebook',
 'musalbas/heartbleed-masstest',
 'cloudant/bigcouch',
 'ShopRunner/jupyter-notify',
 'marshall/logcat-color',
 'dialogflow/dialogflow-python-client',
 'toxinu/Sublimall',
 'appnexus/pyrobuf',
 'turicas/covid19-br',
 'asweigart/my_first_tic_tac_toe',
 'samgiles/slumber',
 'MSiam/TFSegmentation',
 'alegonz/baikal',
 'zorkian/nagios-api',
 'alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras',
 'alexa-pi/AlexaPiDEPRECATED',
 'hangoutsbot/hangoutsbot',
 'musalbas/heartbleed-masstest',
 'cloudant/bigcouch'
]
         
         
         
headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)

    
def get_dataFrame():
    df = pd.read_json('data.json')
    return df
