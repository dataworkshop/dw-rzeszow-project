import logging
import falcon

logging.basicConfig(level=logging.INFO)


class TestResource(object):
    def on_get(self, req, res):
        """Handles all GET requests."""
        print(str(req.media))
        print(str(type(req.media)))
        logging.info("received a GET")
        res.status = falcon.HTTP_200  # This is the default status
        res.body = 'This is me, Falcon, serving a resource!'


# Create the Falcon application object
app = falcon.API()

# Instantiate the TestResource class
test_resource = TestResource()

# Add a route to serve the resource
app.add_route('/test', test_resource)
