"""API documentation views for WallStreetBots.

Provides:
- Interactive API documentation (Swagger UI)
- OpenAPI schema endpoint
- ReDoc alternative documentation
"""

import json

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .schema import get_openapi_schema


@csrf_exempt
@require_http_methods(["GET"])
def openapi_schema_view(request):
    """Return the OpenAPI schema as JSON."""
    schema = get_openapi_schema()
    return JsonResponse(schema, json_dumps_params={"indent": 2})


@csrf_exempt
@require_http_methods(["GET"])
def api_docs_view(request):
    """Render Swagger UI for interactive API documentation."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WallStreetBots API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info .title { color: #1a1a2e; }
        .swagger-ui .info .description p { color: #333; }
        .swagger-ui .scheme-container { background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        /* Custom header */
        .custom-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .custom-header h1 { margin: 0; font-size: 24px; font-weight: 600; }
        .custom-header p { margin: 5px 0 0 0; opacity: 0.8; font-size: 14px; }
        .version-badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="custom-header">
        <div>
            <h1>WallStreetBots API</h1>
            <p>Trading Platform REST API Documentation</p>
        </div>
        <span class="version-badge">v1.0.0</span>
    </div>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "/api/docs/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                displayRequestDuration: true,
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                defaultModelsExpandDepth: 2,
                defaultModelExpandDepth: 2,
                docExpansion: "list",
                syntaxHighlight: {
                    activate: true,
                    theme: "monokai"
                }
            });
            window.ui = ui;
        };
    </script>
</body>
</html>
"""
    return HttpResponse(html, content_type="text/html")


@csrf_exempt
@require_http_methods(["GET"])
def redoc_view(request):
    """Render ReDoc for alternative API documentation."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WallStreetBots API Documentation - ReDoc</title>
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; }
    </style>
</head>
<body>
    <redoc spec-url="/api/docs/openapi.json"></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
</body>
</html>
"""
    return HttpResponse(html, content_type="text/html")
