/**
 * Simple visual test - just checks if chart SVG renders
 * Faster and simpler than full Puppeteer test
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5000';

async function quickTest() {
    console.log('Starting quick chart render test...\n');

    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    try {
        // Enable console logging
        page.on('console', msg => console.log('Browser:', msg.text()));
        page.on('pageerror', error => console.log('Page error:', error.message));

        // Load chart page
        console.log('Loading chart page...');
        await page.goto(`${BASE_URL}/chart/BTCUSDT/1h`, {
            waitUntil: 'networkidle2',
            timeout: 20000
        });
        console.log('✅ Page loaded');

        // Wait for Plotly
        await page.waitForFunction(() => typeof Plotly !== 'undefined', { timeout: 10000 });
        console.log('✅ Plotly loaded');

        // Wait for chart div
        await page.waitForSelector('#chart', { timeout: 5000 });
        console.log('✅ Chart div found');

        // Wait longer for chart render
        console.log('Waiting for chart to render...');
        await page.waitForTimeout(8000);

        // Check for SVG
        const hasSVG = await page.evaluate(() => {
            const svg = document.querySelector('#chart svg.main-svg');
            const traces = document.querySelectorAll('#chart .trace').length;
            const candleSelect = document.getElementById('candleCount')?.value;

            return {
                svgExists: svg !== null,
                svgWidth: svg ? svg.getAttribute('width') : null,
                svgHeight: svg ? svg.getAttribute('height') : null,
                traceCount: traces,
                candleCount: candleSelect
            };
        });

        console.log('\n📊 Chart Status:');
        console.log(`   SVG Rendered: ${hasSVG.svgExists ? '✅ Yes' : '❌ No'}`);
        console.log(`   SVG Size: ${hasSVG.svgWidth} x ${hasSVG.svgHeight}`);
        console.log(`   Trace Count: ${hasSVG.traceCount}`);
        console.log(`   Candle Count: ${hasSVG.candleCount}`);

        // Take screenshot
        const screenshotPath = path.join(__dirname, 'quick_test.png');
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.log(`\n📸 Screenshot: ${screenshotPath}`);

        if (hasSVG.svgExists && hasSVG.traceCount >= 6) {
            console.log('\n✅ Chart is rendering correctly!\n');
            await browser.close();
            process.exit(0);
        } else {
            console.log('\n❌ Chart is NOT rendering correctly!\n');
            await browser.close();
            process.exit(1);
        }

    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        await page.screenshot({ path: 'quick_test_error.png', fullPage: true });
        await browser.close();
        process.exit(1);
    }
}

quickTest();
