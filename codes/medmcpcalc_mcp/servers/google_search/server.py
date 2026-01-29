"""
An MCP server for Google Search
"""
import os
import re
import json
import logging
import asyncio
from typing import List, Dict, Tuple, Any
from bs4 import BeautifulSoup, Tag, Comment, NavigableString

import httpx
import click
import markdownify
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Configure constants
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
API_URL = os.environ.get("GOOGLE_API_URL")
API_KEY = os.environ.get("GOOGLE_API_KEY")


async def _search(
        query: str,
        num_results: int = 10,
        timeout: float = 30
) -> List[Dict[str, Any]]:
    """
    Make a POST request to the Google Search.
    """
    if not API_KEY:
        raise ValueError("PI_KEY environment variable is not set")

    # 1. Construct headers (following curl/requests format)
    headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }

    # 2. Construct payload (Serper requires JSON body)
    payload = {
        "q": query,
        "num": num_results,
    }

    async with httpx.AsyncClient() as client:
        try:
            # Note: Changed to POST request, data passed via json=payload
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()

            # Serper's response structure is mainly in the 'organic' field
            organic_results = data.get("organic", [])

            # Format results
            clean_results = []
            for result in organic_results:
                clean_results.append({
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet"),
                    "position": result.get("position"),
                    "date": result.get("date") # Serper sometimes returns dates
                })
            
            return clean_results
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise



def extract_content_from_html(html: str) -> str:
    """
    Clean DOM using BeautifulSoup, then convert to Markdown using markdownify.
    Includes extensive defensive error checking to prevent NoneType errors.
    """
    if not html:
        return "<error>Empty HTML content</error>"

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # ---------------------------------------------------------
        # 1. Remove disruptive tags (Structural Cleaning)
        # ---------------------------------------------------------
        # Use soup.find_all instead of soup([]) to ensure stability
        for tag in soup.find_all(['script', 'style', 'meta', 'link', 'noscript', 'iframe', 'svg', 'symbol', 'defs']):
            if tag: tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            if comment: comment.extract()

        for tag in soup.find_all(['nav', 'footer', 'aside', 'header']):
            if tag: tag.decompose()

        # ---------------------------------------------------------
        # 2. Heuristic cleaning based on class/ID
        # ---------------------------------------------------------
        noise_keywords = [
            'cookie', 'consent', 'banner', 'newsletter', 'popup', 'modal',
            'ad-', 'ads', 'advert', 'promotion', 'sponsor',
            'share', 'social', 'widget', 'sidebar',
            'related', 'recommend', 'suggestion',
            'menu', 'navigation', 'copyright', 'disclaimer'
        ]

        def is_noise_element(tag):
            """Determine if an element is noise, with strict defensive checks"""
            # 1. Basic check: skip if None or is a string
            if tag is None or isinstance(tag, NavigableString):
                return False

            # 2. Check if it has attrs attribute (ensure it's a Tag object)
            if not hasattr(tag, 'attrs') or not hasattr(tag, 'get'):
                return False

            try:
                # 3. Safely get id and class
                # Using tag.attrs.get() is more explicit
                id_attr = tag.attrs.get('id')
                class_attr = tag.attrs.get('class')

                id_list = []
                if isinstance(id_attr, str):
                    id_list = [id_attr]
                elif isinstance(id_attr, list):
                    id_list = id_attr

                class_list = []
                if isinstance(class_attr, list):
                    class_list = class_attr
                elif isinstance(class_attr, str):
                    class_list = [class_attr]

                attrs = id_list + class_list
                attr_str = " ".join([str(a) for a in attrs]).lower()

                if any(k in attr_str for k in noise_keywords):
                    # Safeguard
                    if not any(safe in attr_str for safe in ['main', 'content', 'article', 'post', 'body']):
                        return True
            except Exception:
                # If individual tag check fails, conservatively don't delete it
                return False

            return False

        # When iterating containers, convert to list first to prevent DOM modification during iteration
        for tag in list(soup.find_all(['div', 'section', 'ul', 'ol', 'dl'])):
            if tag and is_noise_element(tag):
                tag.decompose()

        # ---------------------------------------------------------
        # 3. Remove empty tags (Empty Tag Removal)
        # ---------------------------------------------------------
        for tag in list(soup.find_all(['div', 'p', 'span', 'section'])):
            if tag is None: continue

            try:
                # get_text may not exist on some special nodes
                has_text = False
                if hasattr(tag, 'get_text'):
                    has_text = bool(tag.get_text(strip=True))

                # find may return None, or tag itself may not have find method
                has_children = False
                if hasattr(tag, 'find'):
                    found = tag.find(['img', 'input', 'button', 'select', 'textarea'])
                    has_children = bool(found)

                if not has_text and not has_children:
                    tag.decompose()
            except Exception:
                continue

        # ---------------------------------------------------------
        # 4. Convert to Markdown
        # ---------------------------------------------------------
        content = markdownify.markdownify(
            str(soup),
            heading_style="ATX"
        )

        # ---------------------------------------------------------
        # 5. Post-processing
        # ---------------------------------------------------------
        noise_patterns = [
            r"^\[#+\s*Support\]",             # Filter [#### Support]
            r"^\[#+\s*Login\]",               # Filter [#### Login]
            r"^\[#+\s*Sign\s*Up\]",           # Filter [#### SignUp] or [#### Sign Up]
            r"^\[#+\s*Become a Contributor\]",# Filter [#### Become a Contributor]
            # r"^\[#+\s*All Calculators\]",     # Filter [#### All Calculators]
            # r"^\[#+\s*Calculator\]",          # Filter [#### Calculator] (if repeated)
            
            r"To improve your experience, we \(and our partners\)",
            r"Our website may use these cookies",
            r"Measure the audience of the advertising",
            r"Display personalized ads",
            r"Personalize our editorial content",
            r"Allow you to share content",
            r"Send you advertising",
            r"Manage Preferences",
            r"Accept All",
            r"Reject All",
            r"Close Cookie Preferences",
            r"\[Privacy Policy\]",
            r"^Calc Function",
            r"^Diagnosis$",
            r"^Rule Out$",
            r"^Prognosis$",
            r"^Formula$",
            r"^Treatment$",
            r"^Algorithm$",
            r"^Condition$",
            r"^Specialty$",
            r"^Chief Complaint$",
            r"^Organ System$",
            r"^Select\.\.\.$",
            r"^Log in",
            r"^Sign up",
            r"^Copy Results",
            r"Also from MDCalc\.\.\.",
            r"You might be interested in\.\.\.",
            r"Partner Content",
            r"Emergency Medicine Practice Extra",
            r"Have feedback about this calculator\?",
            r"Get up to 10 hours of stroke CME",
            r"^BODY TEXT HERE"
        ]
        
        lines = []
        for line in content.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            is_noise = False
            for pattern in noise_patterns:
                if re.search(pattern, stripped_line, re.IGNORECASE):
                    is_noise = True
                    break
            
            if not is_noise:
                lines.append(stripped_line)

        cleaned_content = "\n".join(lines)
        
        if not cleaned_content:
            return "<error>Page content appeared empty after conversion.</error>"
        
        return cleaned_content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<error>Failed to extract content: {str(e)}</error>"
    

async def fetch_url_dynamic(
    url: str, user_agent: str, force_raw: bool = False
) -> Tuple[str, str]:
    """
    Use Playwright to fetch dynamically rendered page content and perform noise reduction in the browser.
    Returns: (content, prefix_message)
    """

    # Launch Playwright
    async with async_playwright() as p:
        try:
            # Launch headless browser
            browser = await p.chromium.launch(headless=True)

            # Create context
            context = await browser.new_context(
                user_agent=user_agent,
                # Block image and font loading to speed up and reduce traffic
                permissions=['geolocation'],
            )

            # Route interception can also be set here to block images, styles, and tracking scripts
            await context.route("**/*.{png,jpg,jpeg,gif,css,woff,woff2}", lambda route: route.abort())

            page = await context.new_page()

            logger.info(f"Fetching URL: {url}")

            # Visit URL
            # domcontentloaded is faster than networkidle and usually sufficient for text
            # If content is asynchronously delayed, keep networkidle
            response = await page.goto(url, timeout=30000, wait_until="networkidle")
            
            if not response:
                await browser.close()
                return f"<error>Failed to get response from {url}</error>", ""
            
            if response.status >= 400:
                await browser.close()
                return f"<error>Failed to fetch {url} - status code {response.status}</error>", ""

            # ---------------------------------------------------------
            # Key step: Execute cleaning logic inside the browser (DOM Cleaning)
            # This step removes all CSS, JS, SVG icons, hidden elements, etc.
            # ---------------------------------------------------------
            await page.evaluate("""() => {
                // 1. Remove disruptive tags (CSS, JS, embedded data, media)
                const tagsToRemove = [
                    'script', 'style', 'noscript', 'link[rel="stylesheet"]',
                    'svg', 'iframe', 'object', 'embed',
                    'meta', 'button', 'input', 'select', 'textarea' // Remove form elements (depends on needs, MDCalc may need to keep labels)
                ];

                document.querySelectorAll(tagsToRemove.join(',')).forEach(el => el.remove());

                // 2. Remove class names/IDs that usually contain ads or irrelevant content (add based on experience)
                const selectorsToRemove = [
                    '#cookie-banner', '.cookie-consent', '.ad-container',
                    '[role="alert"]', 'footer', 'nav'
                ];
                try {
                    document.querySelectorAll(selectorsToRemove.join(',')).forEach(el => el.remove());
                } catch(e) {}

                // 3. Handle images: remove Base64 images, keep meaningful image Alt text
                document.querySelectorAll('img').forEach(img => {
                    // If it's a base64 data image, remove it directly or replace with text
                    if (img.src.startsWith('data:')) {
                        img.remove();
                    }
                });

                // 4. Remove hidden elements (display: none)
                // Note: This is time-consuming, can be skipped if the page is very large
                document.querySelectorAll('*').forEach(el => {
                    if (window.getComputedStyle(el).display === 'none') {
                        el.remove();
                    }
                });
            }""")

            # Get cleaned HTML
            cleaned_html = await page.content()
            await browser.close()

        except Exception as e:
            logger.error(f"Browser fetch failed: {e}")
            return f"<error>Browser fetch failed: {str(e)}</error>", ""

    if force_raw:
        return cleaned_html, "Here is the cleaned HTML content:\n"

    # Convert to Markdown/text
    return extract_content_from_html(cleaned_html), ""

def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server with the specified configuration.
    """
    mcp = FastMCP("google_search")

    # --- Tool 1: Google Search ---
    @mcp.tool()
    async def search(query: str) -> str:
        """
        Execute a Google search and return the top results.

        Args:
            query: The search query string.
        """
        logger.info(f"Received search request: {query}")
        num_results = 10
        try:
            # Execute search
            items = await _search(query=query, num_results=num_results)

            if not items:
                return "No results found."

            # Return formatted JSON string
            return json.dumps(items, ensure_ascii=False, indent=2)
            
        except ValueError as ve:
            return json.dumps({"error": f"Configuration error: {str(ve)}"})
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return json.dumps({"error": f"Search failed: {str(e)}"})

    # --- Tool 2: Web Fetch (Playwright) ---
    @mcp.tool()
    async def fetch(url: str) -> str:
        """
        Fetches a URL using a headless browser (Playwright).
        Supports dynamic content (JS), SPAs, and complex layouts.
        Converts HTML to Markdown for easier reading.
        
        Args:
            url: The URL to fetch.
        """
        logger.info(f"Received fetch request: {url})")

        if not url:
            return "<error>URL is required</error>"

        # Call Playwright logic
        content, prefix = await fetch_url_dynamic(url, DEFAULT_USER_AGENT)

        # Handle content truncation (Pagination Logic)
        original_length = len(content)

        # Check if content starts with error tag, if so, return directly
        if content.startswith("<error>"):
            return content

        start_index = 0
        max_length = len(content)
        if start_index >= original_length:
            return f"<error>No more content available. Total length was {original_length}.</error>"
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            return "<error>No content extracted.</error>"
            
        final_output = truncated_content
        actual_content_length = len(truncated_content)
        remaining_content = original_length - (start_index + actual_content_length)

        # If there's remaining content, add a prompt
        if actual_content_length == max_length and remaining_content > 0:
            next_start = start_index + actual_content_length
            final_output += (
                f"\n\n<system_message>\n"
                f"Content truncated. Call the fetch tool again with start_index={next_start} "
                f"to get the next part.\n"
                f"</system_message>"
            )

        return f"{prefix}Contents of {url}:\n\n{final_output}"
    
    return mcp


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default=8000, help="Port to listen on for SSE")
def main(transport: str, port: int):
    """
    Starts the initialized MCP server.
    """
    logger.info(f"Starting the MCP server on port {port} with transport {transport}")
    
    mcp = build_server(port)
    
    if transport == "sse":
        mcp.run(transport="sse", port=port)
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()