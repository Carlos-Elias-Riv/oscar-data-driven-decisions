#!/usr/bin/env node

/**
 * MCP Server for Puppeteer-based IMDb scraping
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import puppeteer from "puppeteer";

const server = new Server(
  {
    name: "imdb-puppeteer-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Global browser instance
let browser = null;

async function getBrowser() {
  if (!browser) {
    browser = await puppeteer.launch({
      headless: true,
      executablePath: '/usr/bin/google-chrome',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-blink-features=AutomationControlled'
      ]
    });
  }
  return browser;
}

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "search_imdb_movie",
        description: "Search for a movie on IMDb and extract its data",
        inputSchema: {
          type: "object",
          properties: {
            title: {
              type: "string",
              description: "Movie title to search for"
            },
            year: {
              type: "string",
              description: "Optional year to help narrow down the search"
            }
          },
          required: ["title"]
        }
      },
      {
        name: "get_imdb_page",
        description: "Navigate to an IMDb URL and extract data",
        inputSchema: {
          type: "object",
          properties: {
            url: {
              type: "string",
              description: "IMDb URL to visit"
            }
          },
          required: ["url"]
        }
      }
    ]
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    if (name === "search_imdb_movie") {
      const { title, year } = args;
      const searchQuery = year ? `${title} ${year}` : title;
      const searchUrl = `https://www.imdb.com/find/?q=${encodeURIComponent(searchQuery)}&s=tt`;

      const browser = await getBrowser();
      const page = await browser.newPage();

      try {
        // Set user agent to avoid detection
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');

        // Navigate to search page
        await page.goto(searchUrl, { waitUntil: 'networkidle0', timeout: 30000 });

        // Wait a bit for dynamic content
        await page.waitForTimeout(2000);

        // Try to find the first result
        const firstResult = await page.$('li.ipc-metadata-list-summary-item');

        if (!firstResult) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  found: false,
                  error: "No results found",
                  url_tried: searchUrl
                })
              }
            ]
          };
        }

        // Get the movie link
        const movieLink = await firstResult.$eval('a.ipc-metadata-list-summary-item__t', el => el.href);

        // Navigate to movie page
        await page.goto(movieLink, { waitUntil: 'networkidle0', timeout: 30000 });
        await page.waitForTimeout(2000);

        // Extract data
        const movieData = await page.evaluate(() => {
          const data = {
            found: true,
            imdb_url: window.location.href
          };

          // Extract rating
          try {
            const ratingElem = document.querySelector('[data-testid="hero-rating-bar__aggregate-rating__score"] span');
            data.rating = ratingElem ? ratingElem.textContent.trim() : '';
          } catch (e) {
            data.rating = '';
          }

          // Extract year
          try {
            const yearElem = document.querySelector('[data-testid="hero-title-block__metadata"] li a');
            data.year = yearElem ? yearElem.textContent.trim() : '';
          } catch (e) {
            data.year = '';
          }

          // Extract metascore
          try {
            const metascoreElem = document.querySelector('[data-testid="metacritic-score-box"] span');
            data.metascore = metascoreElem ? metascoreElem.textContent.trim() : '';
          } catch (e) {
            data.metascore = '';
          }

          // Extract user reviews count
          try {
            const reviewsElem = document.querySelector('[data-testid="hero-rating-bar__user-rating__score"]');
            if (reviewsElem) {
              const parentDiv = reviewsElem.closest('div');
              const countDiv = parentDiv ? parentDiv.querySelector('div[class*="sc"]') : null;
              data.user_reviews = countDiv ? countDiv.textContent.trim() : '';
            } else {
              data.user_reviews = '';
            }
          } catch (e) {
            data.user_reviews = '';
          }

          // Extract cast
          try {
            const castSection = document.querySelector('[data-testid="title-cast"]');
            if (castSection) {
              const castItems = castSection.querySelectorAll('[data-testid="title-cast-item__actor"]');
              const castNames = Array.from(castItems).slice(0, 3).map(item => item.textContent.trim());
              data.stars = JSON.stringify(castNames);
              data.cast_number = castItems.length;
            } else {
              data.stars = '[]';
              data.cast_number = 0;
            }
          } catch (e) {
            data.stars = '[]';
            data.cast_number = 0;
          }

          // Extract runtime
          try {
            const runtimeElem = document.querySelector('[data-testid="hero-title-block__metadata"] li:nth-child(3)');
            data.runtime = runtimeElem ? runtimeElem.textContent.trim() : '';
          } catch (e) {
            data.runtime = '';
          }

          return data;
        });

        movieData.url_tried = searchUrl;
        movieData.url_used = movieLink;

        await page.close();

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(movieData, null, 2)
            }
          ]
        };

      } catch (error) {
        await page.close();
        throw error;
      }

    } else if (name === "get_imdb_page") {
      const { url } = args;

      const browser = await getBrowser();
      const page = await browser.newPage();

      try {
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
        await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
        await page.waitForTimeout(2000);

        const content = await page.content();
        await page.close();

        return {
          content: [
            {
              type: "text",
              text: content.substring(0, 10000) // Limit to first 10000 chars
            }
          ]
        };

      } catch (error) {
        await page.close();
        throw error;
      }
    }

  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            found: false,
            error: error.message
          })
        }
      ],
      isError: true
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("IMDb Puppeteer MCP Server running on stdio");
}

// Handle cleanup
process.on('SIGINT', async () => {
  if (browser) {
    await browser.close();
  }
  process.exit(0);
});

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
