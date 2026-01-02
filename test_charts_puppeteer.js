/**
 * Puppeteer test to verify Flask app charts are rendering
 *
 * Prerequisites:
 *   npm install puppeteer
 *
 * Usage:
 *   1. Start Flask app: .venv\Scripts\python.exe app.py
 *   2. Run this test: node test_charts_puppeteer.js
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5000';
const SCREENSHOT_DIR = path.join(__dirname, 'screenshots');

// Ensure screenshot directory exists
if (!fs.existsSync(SCREENSHOT_DIR)) {
    fs.mkdirSync(SCREENSHOT_DIR);
}

async function testDashboard(browser) {
    console.log('\n📊 Testing Dashboard...');
    const page = await browser.newPage();

    try {
        await page.setViewport({ width: 1920, height: 1080 });

        // Navigate to dashboard
        await page.goto(BASE_URL, { waitUntil: 'networkidle2', timeout: 10000 });

        // Wait for data cards to load
        await page.waitForSelector('.data-card', { timeout: 5000 });

        // Get number of data cards
        const cardCount = await page.$$eval('.data-card', cards => cards.length);
        console.log(`   ✅ Dashboard loaded with ${cardCount} data card(s)`);

        // Take screenshot
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'dashboard.png'),
            fullPage: true
        });
        console.log(`   📸 Screenshot saved: dashboard.png`);

        // Get first card info
        const firstCard = await page.$eval('.data-card', card => {
            const symbol = card.querySelector('.symbol')?.textContent;
            const timeframe = card.querySelector('.timeframe')?.textContent;
            return { symbol, timeframe };
        });

        console.log(`   📈 First card: ${firstCard.symbol} - ${firstCard.timeframe}`);

        await page.close();
        return firstCard;

    } catch (error) {
        console.log(`   ❌ Dashboard test failed: ${error.message}`);
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'dashboard_error.png'),
            fullPage: true
        });
        await page.close();
        throw error;
    }
}

async function testChart(browser, symbol, timeframe) {
    console.log(`\n📈 Testing Chart: ${symbol} ${timeframe}...`);
    const page = await browser.newPage();

    try {
        await page.setViewport({ width: 1920, height: 1080 });

        // Enable console logging from page
        page.on('console', msg => {
            const type = msg.type();
            const text = msg.text();
            if (type === 'error') {
                console.log(`   ⚠️  Browser error: ${text}`);
            } else if (type === 'warning') {
                console.log(`   ⚠️  Browser warning: ${text}`);
            }
        });

        // Capture page errors
        page.on('pageerror', error => {
            console.log(`   ❌ Page error: ${error.message}`);
        });

        // Navigate to chart page
        const chartUrl = `${BASE_URL}/chart/${symbol}/${timeframe}`;
        console.log(`   Loading: ${chartUrl}`);
        await page.goto(chartUrl, { waitUntil: 'networkidle2', timeout: 15000 });

        // Wait for Plotly to load
        await page.waitForFunction(() => typeof Plotly !== 'undefined', { timeout: 5000 });
        console.log('   ✅ Plotly.js loaded');

        // Wait for chart div to exist
        await page.waitForSelector('#chart', { timeout: 5000 });

        // Wait for chart to render (give it more time)
        await page.waitForTimeout(5000);

        // Check if chart is rendered
        const chartExists = await page.evaluate(() => {
            const chartDiv = document.getElementById('chart');
            if (!chartDiv) return false;

            // Check if Plotly has rendered - look for svg element
            const svg = chartDiv.querySelector('svg.main-svg');
            const plotlyDiv = chartDiv.querySelector('.plotly');

            return svg !== null || plotlyDiv !== null;
        });

        if (!chartExists) {
            console.log('   ❌ Chart div not found or Plotly not rendered');
            throw new Error('Chart not rendered');
        }

        console.log('   ✅ Chart rendered successfully');

        // Get chart information
        const chartInfo = await page.evaluate(() => {
            const chartDiv = document.getElementById('chart');

            // Count traces by looking for plotly trace elements
            let traces = 0;
            try {
                // Try to get trace count from SVG
                const svgTraces = chartDiv.querySelectorAll('.trace');
                traces = svgTraces.length;

                // Fallback: try to get from plotly data
                if (traces === 0) {
                    const plotlyDiv = chartDiv.querySelector('.js-plotly-plot');
                    if (plotlyDiv && plotlyDiv.data) {
                        traces = plotlyDiv.data.length;
                    }
                }
            } catch (e) {
                console.error('Error counting traces:', e);
            }

            // Get selected candle count
            const candleSelect = document.getElementById('candleCount');
            const candleCount = candleSelect ? candleSelect.value : 'unknown';

            // Get price info
            const priceOpen = document.getElementById('priceOpen')?.textContent;
            const priceClose = document.getElementById('priceClose')?.textContent;

            return {
                traces,
                candleCount,
                priceOpen,
                priceClose
            };
        });

        console.log(`   📊 Chart traces: ${chartInfo.traces}`);
        console.log(`   🕯️  Candles displayed: ${chartInfo.candleCount}`);
        console.log(`   💰 Price - Open: ${chartInfo.priceOpen}, Close: ${chartInfo.priceClose}`);

        // Verify we have expected number of traces (1 candlestick + 1 volume + 4 stochastics = 6)
        if (chartInfo.traces < 6) {
            console.log(`   ⚠️  Warning: Expected 6+ traces, got ${chartInfo.traces}`);
        } else {
            console.log(`   ✅ All chart panels rendered (${chartInfo.traces} traces)`);
        }

        // Take screenshot of full page
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, `chart_${symbol}_${timeframe}.png`),
            fullPage: true
        });
        console.log(`   📸 Screenshot saved: chart_${symbol}_${timeframe}.png`);

        // Test zoom functionality
        console.log('   🔍 Testing zoom...');
        try {
            await page.evaluate(() => {
                const chartDiv = document.getElementById('chart');
                const plotlyDiv = chartDiv.querySelector('.js-plotly-plot, .plotly');
                if (plotlyDiv && window.Plotly) {
                    // Simulate zoom by updating layout
                    window.Plotly.relayout(plotlyDiv, {
                        'xaxis.autorange': false,
                        'xaxis.range': [10, 50]  // Zoom to candles 10-50
                    }).catch(err => {
                        console.log('Zoom error (non-critical):', err.message);
                    });
                }
            });

            await page.waitForTimeout(2000);
            await page.screenshot({
                path: path.join(SCREENSHOT_DIR, `chart_${symbol}_${timeframe}_zoomed.png`),
                fullPage: true
            });
            console.log('   ✅ Zoom test completed');
        } catch (zoomError) {
            console.log('   ⚠️  Zoom test skipped (non-critical):', zoomError.message);
        }

        await page.close();
        return true;

    } catch (error) {
        console.log(`   ❌ Chart test failed: ${error.message}`);
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, `chart_${symbol}_${timeframe}_error.png`),
            fullPage: true
        });
        await page.close();
        throw error;
    }
}

async function testAPI(browser) {
    console.log('\n🔌 Testing API Endpoints...');
    const page = await browser.newPage();

    try {
        // Test /api/files
        console.log('   Testing /api/files...');
        await page.goto(`${BASE_URL}/api/files`, { waitUntil: 'networkidle2' });
        const filesData = await page.evaluate(() => JSON.parse(document.body.textContent));
        console.log(`   ✅ /api/files returned ${filesData.files.length} file(s)`);

        if (filesData.files.length > 0) {
            const firstFile = filesData.files[0];
            const symbol = firstFile.symbol;
            const timeframe = firstFile.timeframe;

            // Test /api/latest
            console.log(`   Testing /api/latest/${symbol}/${timeframe}...`);
            await page.goto(`${BASE_URL}/api/latest/${symbol}/${timeframe}`, { waitUntil: 'networkidle2' });
            const latestData = await page.evaluate(() => JSON.parse(document.body.textContent));
            console.log(`   ✅ Latest price: ${latestData.close}`);

            // Test /api/chart (just check if it returns JSON)
            console.log(`   Testing /api/chart/${symbol}/${timeframe}?candles=100...`);
            await page.goto(`${BASE_URL}/api/chart/${symbol}/${timeframe}?candles=100`, { waitUntil: 'networkidle2' });
            const chartData = await page.evaluate(() => {
                const text = document.body.textContent;
                try {
                    const json = JSON.parse(text);
                    return {
                        hasData: json.data && json.data.length > 0,
                        hasLayout: !!json.layout,
                        dataLength: json.data ? json.data.length : 0
                    };
                } catch (e) {
                    return { error: e.message };
                }
            });

            if (chartData.error) {
                console.log(`   ❌ Chart API error: ${chartData.error}`);
            } else {
                console.log(`   ✅ Chart data: ${chartData.dataLength} traces, layout: ${chartData.hasLayout}`);
            }
        }

        await page.close();
        return true;

    } catch (error) {
        console.log(`   ❌ API test failed: ${error.message}`);
        await page.close();
        throw error;
    }
}

async function main() {
    console.log('='.repeat(60));
    console.log('Flask Web Monitor - Puppeteer Chart Test');
    console.log('='.repeat(60));
    console.log(`Base URL: ${BASE_URL}`);
    console.log(`Screenshots: ${SCREENSHOT_DIR}`);

    let browser;

    try {
        // Launch browser
        console.log('\n🚀 Launching browser...');
        browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        console.log('✅ Browser launched');

        // Test dashboard
        const firstCard = await testDashboard(browser);

        // Test API
        await testAPI(browser);

        // Test chart if we have data
        if (firstCard && firstCard.symbol && firstCard.timeframe) {
            await testChart(browser, firstCard.symbol, firstCard.timeframe);
        }

        console.log('\n' + '='.repeat(60));
        console.log('✅ All tests completed successfully!');
        console.log('='.repeat(60));
        console.log(`\n📁 Screenshots saved to: ${SCREENSHOT_DIR}`);
        console.log('\nScreenshots:');
        const screenshots = fs.readdirSync(SCREENSHOT_DIR).filter(f => f.endsWith('.png'));
        screenshots.forEach(file => console.log(`   - ${file}`));

        process.exit(0);

    } catch (error) {
        console.error('\n' + '='.repeat(60));
        console.error('❌ Test failed:', error.message);
        console.error('='.repeat(60));
        process.exit(1);

    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

// Run tests
main();
