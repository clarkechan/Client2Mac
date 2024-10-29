using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using System.Text;

namespace ClienttoMac_4._0
{
    public class Program
    {
        public static void Main(string[] args)
        {
            CreateHostBuilder(args).Build().Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .UseWindowsService()
                .ConfigureServices((hostContext, services) =>
                {
                    services.AddHostedService<Worker>();
                });
    }

    public class Worker : BackgroundService
    {
        private static SemaphoreSlim _logSemaphore = new SemaphoreSlim(1);
        private FileSystemWatcher _watcher;
        private static string apiUrl = "http://192.168.1.253:8889/api_v1/vision_predictor/product_serial/component/classifier";
        private static string inputPath = @"E:\检测图片";
        private static string outputPath = @"E:\Mac检测图片";
        private static string imageType = "jpg";

        protected override Task ExecuteAsync(CancellationToken stoppingToken)
        {
            ProcessExistingFiles();
            StartMonitoring();
            return Task.CompletedTask;
        }

        private void ProcessExistingFiles()
        {
            foreach (var dateDir in Directory.GetDirectories(inputPath))
            {
                string date = Path.GetFileName(dateDir);

                foreach (var snDir in Directory.GetDirectories(dateDir))
                {
                    string sn = Path.GetFileName(snDir);

                    foreach (var imagePath in Directory.GetFiles(snDir, $"*.{imageType}"))
                    {
                        if (!IsImageProcessed(imagePath, date, sn))
                        {
                            ProcessImage(imagePath, date, sn).Wait();
                        }
                    }
                }
            }
        }

        private static bool IsImageProcessed(string imagePath, string date, string sn)
        {
            string jsonFileName = Path.GetFileNameWithoutExtension(imagePath) + "_result.json";
            string ngJsonPath = Path.Combine(outputPath, date, sn, "NG", jsonFileName);
            string okJsonPath = Path.Combine(outputPath, date, sn, "OK", jsonFileName);

            return File.Exists(ngJsonPath) || File.Exists(okJsonPath);
        }

        private void StartMonitoring()
        {
            _watcher = new FileSystemWatcher
            {
                Path = inputPath,
                IncludeSubdirectories = true,
                Filter = $"*.{imageType}",
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite
            };

            _watcher.Created += OnNewImageCreated;
            _watcher.EnableRaisingEvents = true;
        }

        private async void OnNewImageCreated(object sender, FileSystemEventArgs e)
        {
            string imagePath = e.FullPath;
            DirectoryInfo snDirInfo = Directory.GetParent(imagePath);
            DirectoryInfo dateDirInfo = snDirInfo?.Parent;

            if (dateDirInfo == null || snDirInfo == null)
            {
                return;
            }

            string sn = snDirInfo.Name;
            string date = dateDirInfo.Name;

            if (!IsImageProcessed(imagePath, date, sn))
            {
                await ProcessImage(imagePath, date, sn);
            }
        }

        private static async Task ProcessImage(string imagePath, string date, string sn)
        {
            string logFilePath = Path.Combine(outputPath, "log.txt");

            try
            {
                await _logSemaphore.WaitAsync();

                using (StreamWriter logWriter = new StreamWriter(logFilePath, true))
                {
                    logWriter.WriteLine($"{DateTime.Now}: 处理新图片 - {imagePath}");

                    try
                    {
                        bool fileReady = false;
                        while (!fileReady)
                        {
                            try
                            {
                                using (FileStream fs = File.Open(imagePath, FileMode.Open, FileAccess.Read, FileShare.None))
                                {
                                    fileReady = true;
                                }
                            }
                            catch (IOException)
                            {
                                Thread.Sleep(500);
                            }
                        }

                        string base64Image = Convert.ToBase64String(File.ReadAllBytes(imagePath));

                        var requestBody = new
                        {
                            input_meta = new InputMeta
                            {
                                vendor_name = "Dinnar",
                                aoi_hardware_version = "0.1.1",
                                aoi_hardware_config = "ClarkeChan",
                                aoi_software_version = "0.2.1",
                                aoi_software_config = "ClarkeChan"
                            },
                            input_data = new List<InputDataTemplate>
                            {
                                new InputDataTemplate
                                {
                                    type = "file_base64",
                                    meta = new FileMeta
                                    {
                                        filename = Path.GetFileName(imagePath),
                                        filetype = imageType,
                                        timestamp = DateTimeOffset.Now.ToUnixTimeSeconds()
                                    },
                                    content = base64Image
                                }
                            }
                        };

                        string jsonResponse = await SendPostRequest(apiUrl, requestBody);
                        logWriter.WriteLine($"{DateTime.Now}: 图片 {imagePath} 推理成功");

                        using (JsonDocument doc = JsonDocument.Parse(jsonResponse))
                        {
                            var predictResults = doc.RootElement.GetProperty("predict_result_data").GetProperty("predict_results");

                            foreach (JsonElement result in predictResults.EnumerateArray())
                            {
                                double predictedScore = result.GetProperty("meta").GetProperty("predicted_score").GetDouble();
                                string destinationDir;

                                if (predictedScore > 0.5)
                                {
                                    destinationDir = Path.Combine(outputPath, date, sn, "NG");
                                }
                                else
                                {
                                    destinationDir = Path.Combine(outputPath, date, sn, "OK");
                                }

                                Directory.CreateDirectory(destinationDir);

                                string destinationPath = Path.Combine(destinationDir, Path.GetFileName(imagePath));
                                File.Copy(imagePath, destinationPath, true);

                                string jsonFilePath = Path.Combine(destinationDir, Path.GetFileNameWithoutExtension(imagePath) + "_result.json");

                                using (StreamWriter writer = new StreamWriter(jsonFilePath))
                                {
                                    await writer.WriteAsync(result.GetRawText());
                                }

                                logWriter.WriteLine($"{DateTime.Now}: 图片 {imagePath} 被归为 {(predictedScore > 0.5 ? "NG" : "OK")} 并保存结果到 {jsonFilePath}");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        logWriter.WriteLine($"{DateTime.Now}: 处理图片 {imagePath} 时出错 - {ex.Message}");
                    }
                }
            }
            finally
            {
                _logSemaphore.Release();
            }
        }

        private static async Task<string> SendPostRequest(string apiUrl, object requestBody)
        {
            using (HttpClient client = new HttpClient())
            {
                string jsonRequest = JsonSerializer.Serialize(requestBody);
                var content = new StringContent(jsonRequest, Encoding.UTF8, "application/json");

                HttpResponseMessage response = await client.PostAsync(apiUrl, content);
                response.EnsureSuccessStatusCode();

                return await response.Content.ReadAsStringAsync();
            }
        }
    }

    #region 数据格式
    public class FileMeta
    {
        public string filename { get; set; }
        public string filetype { get; set; }
        public long timestamp { get; set; }
    }

    public class InputDataTemplate
    {
        public string type { get; set; }
        public FileMeta meta { get; set; }
        public string content { get; set; }
    }

    public class InputMeta
    {
        public string vendor_name { get; set; }
        public string aoi_hardware_version { get; set; }
        public string aoi_hardware_config { get; set; }
        public string aoi_software_version { get; set; }
        public string aoi_software_config { get; set; }
    }
    #endregion
}
