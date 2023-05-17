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

REPOS = ['iconic/SVGInjector',
 'olahol/reactpack',
 'cbrauckmuller/helium',
 'OverZealous/run-sequence',
 'pavelk2/social-feed',
 'jsmreese/moment-duration-format',
 'stolksdorf/Parallaxjs',
 'hxgf/smoke.js',
 'maxzhang/maxzhang.github.com',
 'dylang/grunt-notify',
 'evil-huawei/evil-huawei',
 'mishoo/UglifyJS-old',
 'mgonto/restangular',
 'danialfarid/ng-file-upload',
 'kelektiv/node-cron',
 'fengyuanchen/cropper',
 'kimmobrunfeldt/progressbar.js',
 'ustbhuangyi/vue-analysis',
 'fancyapps/fancybox',
 'vuelidate/vuelidate',
 'mikemintz/react-rethinkdb',
 'stephenplusplus/grunt-wiredep',
 'drduan/minggeJS',
 'borismus/sonicnet.js',
 'dchester/epilogue',
 'joshaven/string_score',
 'wycats/jquery-offline',
 'seaofclouds/tweet',
 'loganfsmyth/babel-plugin-transform-decorators-legacy',
 'soygul/koan',
 'l20n/l20n.js',
 'node-pcap/node_pcap',
 'smalldots/smalldots',
 'filamentgroup/loadJS',
 'hughsk/envify',
 'jboesch/Gritter',
 'markgoodyear/headhesive.js',
 'mjsarfatti/nestedSortable',
 'Armax/Pokemon-GO-node-api',
 'benhowdle89/svgeezy',
 'prose/prose',
 'tbranyen/backbone-boilerplate',
 'angular-translate/angular-translate',
 'jonathantneal/flexibility',
 'rendrjs/rendr',
 'rchipka/node-osmosis',
 'tangbc/vue-virtual-scroll-list',
 'reactjs/react-codemod',
 'square/cube',
 'bebraw/jswiki',
 'vuejs/babel-plugin-transform-vue-jsx',
 'ustwo/ustwo.com-frontend',
 'node-js-libs/node.io',
 'PaulTaykalo/objc-dependency-visualizer',
 'asm-js/validator',
 'andrewelkins/Laravel-4-Bootstrap-Starter-Site',
 'worrydream/Tangle',
 'gitsummore/nile.js',
 'hypercore-protocol/hyperdrive',
 'maxogden/cool-ascii-faces',
 'shama/gaze',
 'mbostock/stack',
 'taptapship/wiredep',
 'sergeyksv/tingodb',
 'request/request-promise-native',
 'MeoMix/StreamusChromeExtension',
 'domenic/sinon-chai',
 'Treesaver/treesaver',
 'kmalakoff/knockback',
 'wking-io/elm-live',
 'gpbl/isomorphic500',
 'walmartlabs/thorax',
 'dominictarr/scuttlebutt',
 'game-helper/weixin',
 'parse-community/ParseReact',
 'simplewebrtc/signalmaster',
 'hilongjw/vue-zhihu-daily',
 'kenberkeley/vue-demo',
 'zackargyle/service-workers',
 'kirjs/react-highcharts',
 'binci/binci',
 'tors/jquery-fileupload-rails',
 'wp-shortcake/shortcake',
 'dkfbasel/vuex-i18n',
 'reyesr/fullproof',
 'ukupat/trolol',
 'webpack-contrib/bundle-loader',
 'fullscale/elastic.js',
 'substance/data',
 'AppianZ/multi-picker',
 'nate-parrott/Flashlight',
 'yunjey/stargan',
 'hwalsuklee/tensorflow-generative-model-collections',
 'brightmart/albert_zh',
 'bulletmark/libinput-gestures',
 'atlanhq/camelot',
 'gnemoug/distribute_crawler',
 'facebookresearch/MUSE',
 'cysmith/neural-style-tf',
 'adamerose/PandasGUI',
 'slackapi/python-rtmbot',
 'LoyaltyNZ/alchemy-framework',
 'pydanny/cached-property',
 'codekansas/keras-language-modeling',
 'j-bennet/wharfee',
 'kvh/ramp',
 'csurfer/gitsuggest',
 'aurora95/Keras-FCN',
 'InsaneLife/dssm',
 'anishathalye/neural-hash-collider',
 'aurora95/Keras-FCN',
 'p1r06u3/opencanary_web',
 'joschu/cgt',
 'espeed/bulbs',
 'cerndb/dist-keras',
 'gornostal/Modific',
 'ParhamP/altify',
 'bitly/asyncmongo',
 'yasintoy/Slack-Gitsin',
 'flashingpumpkin/django-socialregistration',
 'hangoutsbot/hangoutsbot',
 'musalbas/heartbleed-masstest',
 'cloudant/bigcouch',
 'ShopRunner/jupyter-notify',
 'marshall/logcat-color',
 'dialogflow/dialogflow-python-client',
 'toxinu/Sublimall',
 'appnexus/pyrobuf',
 'turicas/covid19-br',
 'BNMetrics/logme',
 'samgiles/slumber',
 'tomekwojcik/envelopes',
 'zorkian/nagios-api',
 'alexa-pi/AlexaPiDEPRECATED',
 'hangoutsbot/hangoutsbot',
 'musalbas/heartbleed-masstest',
 'cloudant/bigcouch',
 'ShopRunner/jupyter-notify',
 'marshall/logcat-color',
 'dialogflow/dialogflow-python-client',
 'kristianperkins/x_x',
 'j2labs/brubeck',
 'jython/frozen-mirror',
 'hjacobs/kube-downscaler',
 'pycassa/pycassa',
 'taraslayshchuk/es2csv',
 'subho406/OmniNet',
 'brutasse/graphite-api',
 'CongWeilin/mtcnn-caffe',
 '7sDream/pyqart',
 'will8211/unimatrix',
 'viper-framework/viper',
 'learning-at-home/hivemind',
 'fossasia/x-mario-center',
 'appium/python-client',
 'OpnTec/open-spectrometer-hardware',
 'titoBouzout/Dictionaries',
 'JasperSnoek/spearmint',
 'andersbll/deeppy',
 'OpenRCE/sulley',
 'elanmart/cbp-translate',
 'kristovatlas/osx-config-check',
 'HelloGitHub-Team/HelloDjango-blog-tutorial',
 'JustGlowing/minisom',
 'redacted/XKCD-password-generator',
 'pypa/packaging.python.org',
 'PyTables/PyTables',
 'joh/when-changed',
 'IronLanguages/main',
 'iGio90/Dwarf',
 'bueda/tornado-boilerplate',
 'aponxi/sublime-better-coffeescript',
 'mukund109/word-mesh',
 'norbusan/debian-speg',
 'norbusan/debian-pycson',
 'mrkipling/maraschino',
 'mesnilgr/is13',
 'ciscocsirt/malspider',
 'blinktrade/bitex',
 'zomux/deepy',
 'xamarin/XobotOS',
 'andreasschrade/android-design-template',
 'karonl/InDoorSurfaceView',
 'android-cn/android-open-project-demo',
 'aol/micro-server',
 'cundong/ZhihuPaper',
 'binaryfork/Spanny',
 'smanikandan14/Volley-demo',
 'gorbin/ASNE',
 'LinkedInAttic/camus',
 'dodola/android_waterfall',
 'fullstackreact/react-native-firestack',
 'wenmingvs/LogReport',
 'gaugesapp/gauges-android',
 'googlearchive/android-EmojiCompat',
 'maoruibin/GankDaily',
 'antonyt/InfiniteViewPager',
 'txusballesteros/fit-chart',
 'thiagolocatelli/android-uitableview',
 'ym6745476/andbase',
 'fullstackreact/react-native-firestack',
 'abel533/guns',
 'wenmingvs/LogReport',
 'gaugesapp/gauges-android',
 'googlearchive/android-EmojiCompat',
 'maoruibin/GankDaily',
 'antonyt/InfiniteViewPager',
 'IntruderShanky/Squint',
 'txusballesteros/fit-chart',
 'thiagolocatelli/android-uitableview',
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
 'dodola/android_waterfall',
 'fujianlian/KLineChart',
 'fullstackreact/react-native-firestack',
 'abel533/guns',
 'wenmingvs/LogReport',
 'gaugesapp/gauges-android',
 'googlearchive/android-EmojiCompat',
 'maoruibin/GankDaily',
 'antonyt/InfiniteViewPager',
 'armzilla/amazon-echo-ha-bridge',
 'LinkedInAttic/datafu',
 'orhanobut/wasp',
 'nekocode/Emojix',
 'agirbal/umongo',
 'zeromq/jzmq',
 'gdpancheng/LoonAndroid',
 'waylife/RedEnvelopeAssistant',
 'Yellow5A5/ActSwitchAnimTool',
 'rorist/android-network-discovery',
 'florent37/LongShadow',
 'agirbal/umongo',
 'zeromq/jzmq',
 'gdpancheng/LoonAndroid',
 'waylife/RedEnvelopeAssistant',
 'yanbober/AvatarLabelView',
 'Yellow5A5/ActSwitchAnimTool',
 'rorist/android-network-discovery',
 'florent37/LongShadow',
 'jacobmoncur/QuiltViewLibrary',
 'wujingchao/MultiCardMenu',
 'Netflix/blitz4j',
 'fafaldo-zz/FABToolbar',
 'Namir233/ZrcListView',
 'gjiazhe/LayoutSwitch',
 'rno/Android-ScrollBarPanel',
 'idic779/monthweekmaterialcalendarview',
 'SimonVT/android-numberpicker',
 'Telenav/NodeFlow',
 'andyb129/FlipsideCamera',
 'KingsMentor/MobileVisionBarcodeScanner',
 'AlbertGrobas/PolygonImageView',
 'H07000223/FlycoBanner_Master',
 'ahorn/android-rss',
 'gitskarios/Gitskarios',
 'bhavesh-hirpara/MultipleImagePick',
 'hougr/SmartisanPull',
 'Subito-it/Masaccio',
 'pavlospt/CircleView',
 'Android500/AwesomeDrawer',
 'oubowu/MarqueeLayoutLibrary'
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
