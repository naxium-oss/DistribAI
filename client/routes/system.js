/** Host system metrics (CPU/RAM/network) API routes. */
'use strict';

function registerSystemRoutes(app, deps) {
    const { si, os } = deps;

    app.get('/api/network/metrics', async (req, res) => {
        try {
            const metrics = {
                latency: null,
                bandwidth: null,
                connectionQuality: 'unknown',
                timestamp: Date.now(),
            };

            const latencyTargets = [
                { host: '8.8.8.8', name: 'Google DNS' },
                { host: '1.1.1.1', name: 'Cloudflare DNS' },
                { host: 'google.com', name: 'Google' },
            ];

            const latencyPromises = latencyTargets.map(async (target) => {
                try {
                    const { exec } = require('child_process');
                    const { promisify } = require('util');
                    const execAsync = promisify(exec);

                    const command = process.platform === 'win32'
                        ? `ping -n 1 ${target.host}`
                        : `ping -c 1 ${target.host}`;

                    const { stdout } = await execAsync(command, { timeout: 5000 });

                    const latencyMatch = stdout.match(/time[=<>](\d+\.?\d*)/i)
                        || stdout.match(/(\d+\.?\d*)\s*ms/);

                    if (latencyMatch) {
                        return {
                            host: target.host,
                            name: target.name,
                            latency: parseFloat(latencyMatch[1]),
                        };
                    }
                } catch (err) {
                    return { host: target.host, name: target.name, latency: null };
                }
            });

            const latencyResults = await Promise.all(latencyPromises);
            const validLatencies = latencyResults.filter((r) => r.latency !== null);

            if (validLatencies.length > 0) {
                metrics.latency = {
                    results: validLatencies,
                    average: validLatencies.reduce((sum, r) => sum + r.latency, 0) / validLatencies.length,
                    best: Math.min(...validLatencies.map((r) => r.latency)),
                    worst: Math.max(...validLatencies.map((r) => r.latency)),
                };

                const avgLatency = metrics.latency.average;
                if (avgLatency < 20) {
                    metrics.connectionQuality = 'excellent';
                } else if (avgLatency < 50) {
                    metrics.connectionQuality = 'good';
                } else if (avgLatency < 100) {
                    metrics.connectionQuality = 'fair';
                } else {
                    metrics.connectionQuality = 'poor';
                }
            }

            try {
                const https = require('https');
                const startTime = Date.now();
                const testDataSize = 1024 * 1024;

                const response = await new Promise((resolve, reject) => {
                    const req2 = https.get(`https://httpbin.org/bytes/${testDataSize}`, (res2) => {
                        let data = '';
                        res2.on('data', (chunk) => { data += chunk; });
                        res2.on('end', () => resolve({ data, statusCode: res2.statusCode }));
                        res2.on('error', reject);
                    });
                    req2.on('error', reject);
                    req2.setTimeout(10000, () => reject(new Error('Timeout')));
                });

                const endTime = Date.now();
                const duration = (endTime - startTime) / 1000;

                if (response.statusCode === 200) {
                    const bandwidthMbps = (testDataSize * 8) / (duration * 1024 * 1024);
                    metrics.bandwidth = {
                        downloadMbps: Math.round(bandwidthMbps * 10) / 10,
                        testDuration: duration,
                        testSize: testDataSize,
                    };
                }
            } catch (err) {
                console.warn('Bandwidth test failed:', err.message);
            }

            res.json(metrics);
        } catch (err) {
            console.error('Network metrics error:', err);
            res.status(500).json({ error: 'Failed to get network metrics' });
        }
    });

    app.get('/api/system', async (req, res) => {
        try {
            const [cpu, mem, graphics, osInfo, network, battery, diskLayout] = await Promise.all([
                si.cpu().catch(() => null),
                si.mem().catch(() => null),
                si.graphics().catch(() => null),
                si.osInfo().catch(() => null),
                si.networkInterfaces().catch(() => null),
                si.battery().catch(() => null),
                si.diskLayout().catch(() => null),
            ]);

            let gpuInfo = {
                model: 'No GPU detected',
                vram: 0,
                vendor: 'Unknown',
                driverVersion: null,
                temperature: null,
                utilization: null,
            };

            if (graphics && graphics.controllers && graphics.controllers.length > 0) {
                const gpuCtrl0 = graphics.controllers.find((c) =>
                    c.vram > 0 || c.memoryTotal > 0 || c.utilizationGpu != null
                ) || graphics.controllers[0];

                if (gpuCtrl0) {
                    gpuInfo = {
                        model: gpuCtrl0.model || 'Unknown GPU',
                        vram: gpuCtrl0.vram || gpuCtrl0.memoryTotal || 0,
                        vendor: gpuCtrl0.vendor || 'Unknown',
                        driverVersion: gpuCtrl0.driverVersion || null,
                        temperature: gpuCtrl0.temperatureGpu || null,
                        utilization: gpuCtrl0.utilizationGpu || null,
                    };
                }
            }

            const cpuInfo = cpu ? {
                cores: cpu.cores || os.cpus().length,
                model: cpu.brand || cpu.manufacturer || 'Unknown',
                speed: cpu.speed || 'Unknown',
                physicalCores: cpu.physicalCores || 'Unknown',
                cache: cpu.cache || 'Unknown',
                family: cpu.family || 'Unknown',
            } : {
                cores: os.cpus().length,
                model: 'Unknown',
                speed: 'Unknown',
                physicalCores: 'Unknown',
                cache: 'Unknown',
                family: 'Unknown',
            };

            const memInfo = mem ? {
                total: Math.round(mem.total / (1024 ** 3) * 10) / 10,
                used: Math.round(mem.used / (1024 ** 3) * 10) / 10,
                free: Math.round(mem.free / (1024 ** 3) * 10) / 10,
                available: Math.round(mem.available / (1024 ** 3) * 10) / 10,
                swapTotal: Math.round((mem.swapTotal || 0) / (1024 ** 3) * 10) / 10,
                swapUsed: Math.round((mem.swapUsed || 0) / (1024 ** 3) * 10) / 10,
            } : {
                total: 0,
                used: 0,
                free: 0,
                available: 0,
                swapTotal: 0,
                swapUsed: 0,
            };

            const networkInfo = network ? {
                interfaces: network.map((iface) => ({
                    name: iface.iface,
                    type: iface.type,
                    speed: iface.speed || null,
                    ip4: iface.ip4 || null,
                    ip6: iface.ip6 || null,
                    mac: iface.mac || null,
                    operstate: iface.operstate || 'unknown',
                })),
                defaultInterface: network.find((iface) => iface.default)?.iface || null,
            } : { interfaces: [], defaultInterface: null };

            const systemHealth = {
                uptime: os.uptime(),
                loadAverage: os.loadavg(),
                platform: os.platform(),
                arch: os.arch(),
                hostname: os.hostname(),
                osType: osInfo?.platform || 'Unknown',
                osRelease: osInfo?.release || 'Unknown',
                osDistro: osInfo?.distro || 'Unknown',
                battery: battery ? {
                    percent: battery.percent,
                    charging: battery.charging,
                    health: battery.health,
                } : null,
            };

            const storageInfo = diskLayout ? {
                disks: diskLayout.map((disk) => ({
                    device: disk.device,
                    type: disk.type,
                    size: Math.round(disk.size / (1024 ** 3) * 10) / 10,
                    vendor: disk.vendor || 'Unknown',
                    name: disk.name || 'Unknown',
                    serialNum: disk.serialNum || 'Unknown',
                })),
            } : { disks: [] };

            res.json({
                cpu: cpuInfo,
                memory: memInfo,
                gpu: gpuInfo,
                network: networkInfo,
                system: systemHealth,
                storage: storageInfo,
                timestamp: Date.now(),
            });
        } catch (err) {
            console.error('System info error:', err);
            res.status(500).json({ error: 'Failed to get system info' });
        }
    });

    app.get('/api/stats', async (req, res) => {
        try {
            const [mem, currentLoad, graphics, cpuTemp] = await Promise.all([
                si.mem(),
                si.currentLoad(),
                si.graphics().catch(() => null),
                si.cpuTemperature().catch(() => null),
            ]);

            const controllers = graphics && graphics.controllers ? graphics.controllers : [];
            const gpuCtrl = controllers.find((c) =>
                c.vram > 0 || c.memoryTotal > 0 || c.utilizationGpu != null
            ) || controllers[0] || null;
            const gpu = gpuCtrl ? {
                model: gpuCtrl.model || null,
                vramTotal: gpuCtrl.vram || gpuCtrl.memoryTotal || null,
                vramUsed: gpuCtrl.vramUsed != null ? gpuCtrl.vramUsed : (gpuCtrl.memoryUsed != null ? gpuCtrl.memoryUsed : null),
                vramFree: gpuCtrl.vramFree != null ? gpuCtrl.vramFree : null,
                utilization: gpuCtrl.utilizationGpu != null ? gpuCtrl.utilizationGpu
                    : gpuCtrl.utilization != null ? gpuCtrl.utilization : null,
                temperature: gpuCtrl.temperatureGpu != null ? gpuCtrl.temperatureGpu
                    : gpuCtrl.temperatureC != null ? gpuCtrl.temperatureC : null,
                fanSpeed: gpuCtrl.fanSpeed != null ? gpuCtrl.fanSpeed : null,
                powerDraw: gpuCtrl.powerDraw != null ? gpuCtrl.powerDraw : null,
                coreClock: gpuCtrl.clockCore != null ? gpuCtrl.clockCore : null,
            } : null;

            res.json({
                memory: {
                    total: Math.round(mem.total / (1024 ** 3) * 10) / 10,
                    used: Math.round(mem.used / (1024 ** 3) * 10) / 10,
                    free: Math.round(mem.free / (1024 ** 3) * 10) / 10,
                    usedPercent: Math.round(mem.used / mem.total * 100),
                },
                cpu: {
                    cores: os.cpus().length,
                    usage: Math.round(currentLoad.currentLoad * 10) / 10,
                    temp: cpuTemp && cpuTemp.main ? cpuTemp.main : null,
                },
                gpu,
            });
        } catch (err) {
            res.status(500).json({ error: 'Failed to get stats' });
        }
    });

    app.get('/api/system/info', async (req, res) => {
        try {
            const [cpu, mem, graphics, osInfo] = await Promise.all([
                si.cpu(),
                si.mem(),
                si.graphics(),
                si.osInfo(),
            ]);

            const gpuCtrl = graphics.controllers?.find((c) =>
                c.vram > 0 || c.memoryTotal > 0
            ) || graphics.controllers?.[0] || null;

            res.json({
                cpu: {
                    manufacturer: cpu.manufacturer,
                    brand: cpu.brand,
                    speed: cpu.speed,
                    cores: cpu.cores,
                    physicalCores: cpu.physicalCores,
                    processors: cpu.processors,
                },
                memory: {
                    total: Math.round(mem.total / (1024 ** 3) * 10) / 10,
                    totalGB: Math.round(mem.total / (1024 ** 3)),
                },
                gpu: gpuCtrl ? {
                    model: gpuCtrl.model,
                    vramMB: gpuCtrl.vram || gpuCtrl.memoryTotal || 0,
                    vramGB: Math.round((gpuCtrl.vram || gpuCtrl.memoryTotal || 0) / 1024 * 10) / 10,
                } : null,
                os: {
                    platform: os.platform(),
                    hostname: os.hostname(),
                    release: osInfo.release,
                    distro: osInfo.distro,
                },
            });
        } catch (err) {
            res.status(500).json({ error: 'Failed to get system info' });
        }
    });
}

module.exports = { registerSystemRoutes };
