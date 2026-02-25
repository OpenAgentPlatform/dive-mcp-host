# OAP MCP Transform Example
How to transform OAP MCP config recived from search API to dive mcp config

## Types
### OAuth2
```bash
curl -XPOST -L https://oaphub.ai/api/v1/user/mcp/batch \
-H 'Authorization: Bearer {CLIENT_KEY}' \
-d '{"mcp_ids": ["244719633527996416"]}'
```
```json
{
    "status": "success",
    "error": null,
    "data": [
        {
            "id": "244719633527996416",
            "name": "Gitea",
            "description": "MCP server for managing Gitea repositories, issues, pull requests, and releases",
            "plan": "BASE",
            ... // None related fields are removed
            "transport": "streamable",
            "url": "https://proxy.oaphub.ai/v1/mcp/244719633527996416",
            "auth_type": "oauth2" // <-- Look here
        }
    ]
}
```

Dive MCP Config:
```json
{
    "mcpServers": {
        "Gitea": {
            "transport": "streamable",
            "url": "https://proxy.oaphub.ai/v1/mcp/244719633527996416",
            "extraData": {
              "oap": {
                  "id": "244719633527996416",
                  "planTag": "base",
                  "description": "MCP server for managing Gitea repositories, issues, pull requests, and releases",
              }
            }
        }
    }
}
```


---
### Header Token
```bash
curl -XPOST -L https://oaphub.ai/api/v1/user/mcp/batch \
-H 'Authorization: Bearer {CLIENT_KEY}' \
-d '{"mcp_ids": ["254864871177322496"]}'
```
```json
{
    "status": "success",
    "error": null,
    "data": [
        {
            "id": "254864871177322496",
            "name": "Bytedance Seedream-4.5",
            "description": "Seedream 4.5: Upgraded Bytedance image model with stronger spatial understanding and world knowledge.",
            "plan": "PRO",
            ... // None related fields are removed
            "transport": "streamable",
            "url": "https://proxy.oaphub.ai/v1/mcp/254864871177322496",
            "auth_type": "header" // <-- Look here
        }
    ]
}
```

Dive MCP Config:
```json
{
    "mcpServers": {
        "Bytedance_Seedream_4_5": {
            "transport": "streamable",
            "url": "https://proxy.oaphub.ai/v1/mcp/254864871177322496",
            "headers": {
                "Authorization": "{CLIENT_KEY}" // <-- CLIENT_KEY in the header
            },
            "extraData": {
              "oap": {
                  "id": "254864871177322496",
                  "planTag": "pro",
                  "description": "Seedream 4.5: Upgraded Bytedance image model with stronger spatial understanding and world knowledge.",
              }
            }
        }
    }
}
```


---
### External endpoint
```bash
curl -XPOST -L https://oaphub.ai/api/v1/user/mcp/batch \
-H 'Authorization: Bearer {CLIENT_KEY}' \
-d '{"mcp_ids": ["269699364035756032"]}'
```
If `external_endpoint` is set, values under `external_endpoint` should be prioritized.  
Dive mcp config should be created based on `external_endpoint`
```json
{
    "status": "success",
    "error": null,
    "data": [
        {
            "id": "269699364035756032",
            "name": "File Uploader",
            "description": "Upload local files to OAPhub (Open Agent Platform) remote storage.  \r\nMainly used in combination with the MCP server that accepts URLs for files.  ",
            "plan": "BASE",
            ... // None related fields are removed
            "transport": "streamable", // <-- Ignore
            "external_endpoint": {
                "command": "npx",
                "args": [
                    "@oaphub\/file-uploader-mcp"
                ],
                "env": {
                    "OAP_CLIENT_KEY": "{{AccessToken}}"
                },
                "protocol": "stdio"
            },
            "auth_type": "header" // <-- Ignore
        }
    ]
}
```

Dive MCP Config:
```json
{
    "mcpServers": {
        "File_Uploader": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "@oaphub/file-uploader-mcp"
            ],
            "env": {
                "OAP_CLIENT_KEY": "{{AccessToken}}"
            },
            "extraData": {
              "oap": {
                  "id": "269699364035756032",
                  "planTag": "base",
                  "description": "Upload local files to OAPhub (Open Agent Platform) remote storage.  \r\nMainly used in combination with the MCP server that accepts URLs for files.  ",
              }
            }
        }
    }
}
```

