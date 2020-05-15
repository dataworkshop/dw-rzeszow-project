# sudokusolver
Sudoku solver falcon framework powered PCF app

Endpoint usage: `http://localhost:8000/sudoku?grid=2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3`

## Launching the app locally (example commands for Ubuntu/Debian)
* install git, python and virtualenv `sudo apt-get install virtualenv`
* create a python 3 virtual env `virtualenv venv -p /usr/bin/python3` (not strictly needed but gives better confidence that app would work in the cloud)
* activate the environment `source venv/bin/activate` (`venv\Scripts\activate.bat` on windows)
* install the requirements `pip install -r requirements.txt` (can also install gunicorn globally)
* run `gunicorn main:app`, you can provide more args `gunicorn -b 0.0.0.0:8000 main:app --reload`
* head over to http://localhost:8000/test (or curl or `http you-url` if you have httpie, highly recommend it)

## Running in Pivotal Cloud Foundry
* install cf cli (command line utils to interact with PCF) https://github.com/cloudfoundry/cli#downloads or https://docs.cloudfoundry.org/cf-cli/install-go-cli.html
* create an account in Pivotal https://run.pivotal.io/
* login to PCF `cf login -a https://api.run.pivotal.io -u user-name-or-email -p pass -s your-pcf-space`
* space name and all other things can be looked up in console app in browser console.run.pivotal.io 
* push the app, since this repo defines `manifest.yml` file with required params it's enough to just do `cf push`
* alternatively provide some explicit params eg. `cf push cf-falcon-app -m 128M --random-route` (route is the app's URL, without random route you may run into conflicts, it is `your-app-name.cfapps.io` by default, you can also create the route manually in console web app)
* head over to console and check the state and url of the app eg. https://falcon-py.cfapps.io/test which should respond with the test message 

### Similar examples
* https://github.com/vchrisb/cf-HelloWorld
* https://github.com/swisscom/cf-sample-app-python
* https://www.digitalocean.com/community/tutorials/how-to-deploy-falcon-web-applications-with-gunicorn-and-nginx-on-ubuntu-16-04
