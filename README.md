# PolarRec

PolarRec is an academic resource recommendation engine, tailor-made for the [Zotero](https://www.zotero.org/) research assistant platform. Given a collection of target academic resources (such as papers of interest to the user), PolarRec utilises publicly available data including authors, conference names, publication dates, citations and citation counts to collect and generate recommended resources for the user.

This code repository implements the PolarRec web API, a lightweight monolithic service with a RESTful API that allows client applications to interact with the PolarRec service. Visit the [PolarRec Zotero Plugin](https://github.com/jyjulianwong/PolarRec-Zotero-Plugin) code repository to learn more about the Zotero plugin / add-on itself, which can be downloaded separately and imported into the Zotero client application. The plugin provides a simple user interface that allows users to interact with the PolarRec service and get recommendations based on items in their Zotero library. 

# Getting started

Before getting started with development, ensure that Python 3 (3.8 or above) and Pip 3 have been installed on your device. The following examples demonstrate the process of compiling the application using a Bash terminal.

Clone the repository onto your local device.
```shell
git clone https://github.com/jyjulianwong/PolarRec.git
cd PolarRec
```

Create a Python virtual environment and install the package dependencies.
```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Set the necessary Flask environment variables for development.
```shell
export FLASK_APP="application.py"
export FLASK_ENV="development"
export FLASK_DEBUG=0
```

For the application to access the IEEE Xplore web API as part of the recommendation search process, an IEEE Xplore API key is necessary. API keys can be applied for by [registering an account with IEEE Xplore](https://developer.ieee.org/member/register). Contact me if assistance related to this issue is required. Set your private API key as an environment variable.
```shell
export PLR_IEEE_XPLORE_API_KEY="myieeexploreapikey"
```

Compile the Flask application and run the local development server.
```shell
python3 -m flask run
```

By default, the server should be accessible at `http://127.0.0.1:5000`. For example, to access the recommendation service, POST requests can be sent to `http://127.0.0.1:5000/recommend`.

# The web API

The publicly deployed web API is accessible at `http://polarrec-env.eba-nzautmta.eu-west-2.elasticbeanstalk.com`. To access the recommendation service, POST requests should be sent to `http://polarrec-env.eba-nzautmta.eu-west-2.elasticbeanstalk.com/recommend`. Academic resources are sent and received in JSON format.

The headers of the POST request should be set as follows.
```json
{
    "Accept": "application/json",
    "Content-Type": "application/json"
}
```

The following is an example of the body of a POST request.
```json
{
    "target_resources": [
        {
            "authors": ["R. Girshick","J. Donahue","T. Darrell","J. Malik"],
            "conference_name": "2014 IEEE Conference on Computer Vision...",
            "conference_location": "Columbus, OH, USA",
            "title": "Rich Feature Hierarchies for Accurate Object...",
            "year": 2014,
            "month": 6,
            "abstract": "Object detection performance, as measured on...",
            "doi": "10.1109/CVPR.2014.81",
            "url": "https://ieeexplore.ieee.org/document/6909475",
            "references": [{...}, {...}, ...]
        }
    ],
    "existing_resources": [
        {
            "authors": ["R. Girshick","J. Donahue","T. Darrell","J. Malik"],
            "conference_name": "2014 IEEE Conference on Computer Vision...",
            "conference_location": "Columbus, OH, USA",
            "title": "Rich Feature Hierarchies for Accurate Object...",
            "year": 2014,
            "month": 6,
            "abstract": "Object detection performance, as measured on...",
            "doi": "10.1109/CVPR.2014.81",
            "url": "https://ieeexplore.ieee.org/document/6909475",
            "references": [{...}, {...}, ...]
        }
    ],
    "filter": {
        "authors": ["R. Girshick","J. Donahue","T. Darrell","J. Malik"],
        "conference_name": "2014 IEEE Conference on Computer Vision..."
    }
}
```

The following is an example of the body of the response received from a POST request.
```json
{
    "processing_time": 2.56,
    "ranked_existing_resources": [
        {
            "authors": ["R. Girshick","J. Donahue","T. Darrell","J. Malik"],
            "conference_name": "2014 IEEE Conference on Computer Vision...",
            "conference_location": "Columbus, OH, USA",
            "title": "Rich Feature Hierarchies for Accurate Object...",
            "year": 2014,
            "month": 6,
            "abstract": "Object detection performance, as measured on...",
            "doi": "10.1109/CVPR.2014.81",
            "url": "https://ieeexplore.ieee.org/document/6909475",
            "references": [{...}, {...}, ...],
            "author_based_ranking": 1,
            "citation_based_ranking": 3,
            "keyword_based_ranking": 2
        }
    ],
    "ranked_database_resources": [
        {
            "authors": ["R. Girshick","J. Donahue","T. Darrell","J. Malik"],
            "conference_name": "2014 IEEE Conference on Computer Vision...",
            "conference_location": "Columbus, OH, USA",
            "title": "Rich Feature Hierarchies for Accurate Object...",
            "year": 2014,
            "month": 6,
            "abstract": "Object detection performance, as measured on...",
            "doi": "10.1109/CVPR.2014.81",
            "url": "https://ieeexplore.ieee.org/document/6909475",
            "references": [{...}, {...}, ...],
            "author_based_ranking": 1,
            "citation_based_ranking": 3,
            "keyword_based_ranking": 2
        }
    ],
    "ranked_citation_resources": [
        {
            "authors": ["R. Girshick","J. Donahue","T. Darrell","J. Malik"],
            "conference_name": "2014 IEEE Conference on Computer Vision...",
            "conference_location": "Columbus, OH, USA",
            "title": "Rich Feature Hierarchies for Accurate Object...",
            "year": 2014,
            "month": 6,
            "abstract": "Object detection performance, as measured on...",
            "doi": "10.1109/CVPR.2014.81",
            "url": "https://ieeexplore.ieee.org/document/6909475",
            "references": [{...}, {...}, ...],
            "author_based_ranking": 1,
            "citation_based_ranking": 3,
            "keyword_based_ranking": 2
        }
    ]
}
```